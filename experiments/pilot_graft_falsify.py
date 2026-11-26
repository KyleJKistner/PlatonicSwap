#!/usr/bin/env python3
"""
pilot_graft_falsify.py — Consensus 4-Test Cross-Family Platonic Grafting Falsification Pilot
============================================================================================

Distilled from independent adversarial reviews (Astra 6.2 Ultra, Fable 5.2 Max, Neon High).
Pair: Donor A = qwen2.5-3b (D=2048, L=36) -> Recipient B = smollm2-1.7b (D=2048, L=24).
Uses CPU float64 for ALL SVD/whitening/filter algebra and float32 on MPS for forward passes.
Reuses cache/pilot1_maps_qwen2.5-3b_to_smollm2-1.7b_l17_9.pt at (17 -> 9) and fits leak-free
full-rank unlabeled entity maps (strictly excluding ALL T0/T1/T2/T3 entities & countries)
for swept layers (11 -> 6), (17 -> 9), (20 -> 11) + post-RMSNorm MLP-input sockets!

4 Modular Sub-Tests (--tests t0,t1,t2,t3):
  - T0: Knowledge & Competence Filter (32 famous landmarks across 10 countries + 24 famous
        people fallback; exact shared-prefix geographic + protected prompts).
  - T1: Forced-Gate Value, Multi-Hop Ripple & Orthogonal Retention Audit across 3 depths
        {25%, 37.5%, 46%} x scales {0.5, 1.0, 2.0, 3.0} comparing 7 vector families:
        native_B_noun_diff, socket_noun_diff, native_B_class_mean_diff,
        socket_class_mean_diff, city_country_sibling_native_B,
        city_country_sibling_socket, norm_matched_random_dir, shuffled_alignment_control.
  - T2: Key Selectivity & Hard-Neighbor Margin (k in {64, 128, 256, 512, full};
        donor P^k raw cosine, donor P^k whitened + r^2-weighted, native-B C^{-1}K MVDR filter;
        6 fit + 6 unseen paraphrases vs hard neighbors + all background tokens) +
        Gate 3 implicit-constant sigma_c audit on B's post-RMSNorm states.
  - T3: Fable's High-Upside Knowledge-Gap Noun Transplant (~150 medium/long-tail entities;
        screen A-right / B-wrong; patch B last_subj with rotated A vector via Pilot 1 10k maps;
        test country + 4 unseen dependent attributes; mandatory controls: different gap entity
        from A, B self round-trip, norm-matched random rotation).

Usage:
  python pilot_graft_falsify.py --tests t0,t1,t2,t3
  python pilot_graft_falsify.py --unit-test          # Fast CPU-only math/data verification (no GPU!)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_and_baseline import (
    CACHE_DIR,
    DATASET_PATH,
    LAB_DIR,
    SEED,
    _greedy_decode_from_prefill,
    check_answer_match,
    get_decoder_layers,
    locate_last_subject_token,
)

OUT_PATH = LAB_DIR / "results_graft_falsify.json"
PILOT1_MAPS_PATH = CACHE_DIR / "pilot1_maps_qwen2.5-3b_to_smollm2-1.7b_l17_9.pt"
PILOT1_UNLAB_PATH = CACHE_DIR / "pilot1_unlab_qwen2.5-3b_to_smollm2-1.7b_l17_9.pt"

DONOR_KEY = "qwen2.5-3b"
RECIP_KEY = "smollm2-1.7b"
# Swept layer pairs (l_A, l_B) corresponding to ~25%, 37.5% (Pilot 1 optimal!), and ~46% of SmolLM2-1.7B (L=24)
SWEPT_LAYER_PAIRS: List[Tuple[int, int]] = [(11, 6), (17, 9), (20, 11)]


# ======================================================================================
# 1. Curated Benchmarks (T0/T1/T2 Landmarks + People Fallback, T3 Knowledge-Gap Pool)
# ======================================================================================

COUNTRIES_INFO: Dict[str, Dict[str, Any]] = {
    "France": {
        "aliases": ["France"],
        "capital": ("Paris", ["Paris"]),
        "currency": ("euro", ["euro", "Euro", "euros"]),
        "language": ("French", ["French"]),
        "continent": ("Europe", ["Europe"]),
    },
    "Japan": {
        "aliases": ["Japan"],
        "capital": ("Tokyo", ["Tokyo"]),
        "currency": ("yen", ["yen", "Japanese yen"]),
        "language": ("Japanese", ["Japanese"]),
        "continent": ("Asia", ["Asia"]),
    },
    "United States": {
        "aliases": ["United States", "the United States", "USA", "US", "America"],
        "capital": ("Washington", ["Washington", "Washington, D.C.", "Washington D.C."]),
        "currency": ("dollar", ["dollar", "US dollar", "U.S. dollar", "United States dollar", "American dollar"]),
        "language": ("English", ["English"]),
        "continent": ("North America", ["North America"]),
    },
    "Italy": {
        "aliases": ["Italy"],
        "capital": ("Rome", ["Rome"]),
        "currency": ("euro", ["euro", "Euro", "euros"]),
        "language": ("Italian", ["Italian"]),
        "continent": ("Europe", ["Europe"]),
    },
    "United Kingdom": {
        "aliases": ["United Kingdom", "the United Kingdom", "UK", "Britain", "England"],
        "capital": ("London", ["London"]),
        "currency": ("pound", ["pound", "British pound", "pound sterling", "sterling"]),
        "language": ("English", ["English"]),
        "continent": ("Europe", ["Europe"]),
    },
    "Germany": {
        "aliases": ["Germany"],
        "capital": ("Berlin", ["Berlin"]),
        "currency": ("euro", ["euro", "Euro", "euros"]),
        "language": ("German", ["German"]),
        "continent": ("Europe", ["Europe"]),
    },
    "India": {
        "aliases": ["India"],
        "capital": ("New Delhi", ["New Delhi", "Delhi"]),
        "currency": ("rupee", ["rupee", "Indian rupee"]),
        "language": ("Hindi", ["Hindi"]),
        "continent": ("Asia", ["Asia"]),
    },
    "Brazil": {
        "aliases": ["Brazil"],
        "capital": ("Brasilia", ["Brasilia", "Brasília"]),
        "currency": ("real", ["real", "Brazilian real"]),
        "language": ("Portuguese", ["Portuguese"]),
        "continent": ("South America", ["South America"]),
    },
    "Russia": {
        "aliases": ["Russia"],
        "capital": ("Moscow", ["Moscow"]),
        "currency": ("ruble", ["ruble", "rouble", "Russian ruble", "Russian rouble"]),
        "language": ("Russian", ["Russian"]),
        "continent": ("Europe", ["Europe", "Asia", "Eurasia"]),
    },
    "China": {
        "aliases": ["China", "the People's Republic of China"],
        "capital": ("Beijing", ["Beijing"]),
        "currency": ("yuan", ["yuan", "renminbi", "Chinese yuan", "RMB"]),
        "language": ("Mandarin", ["Mandarin", "Chinese", "Standard Chinese", "Mandarin Chinese"]),
        "continent": ("Asia", ["Asia"]),
    },
    # Extra countries for T3 knowledge-gap pool
    "Austria": {"aliases": ["Austria"], "capital": ("Vienna", ["Vienna"]), "currency": ("euro", ["euro", "Euro"]), "language": ("German", ["German"]), "continent": ("Europe", ["Europe"])},
    "Bulgaria": {"aliases": ["Bulgaria"], "capital": ("Sofia", ["Sofia"]), "currency": ("lev", ["lev", "Bulgarian lev"]), "language": ("Bulgarian", ["Bulgarian"]), "continent": ("Europe", ["Europe"])},
    "Estonia": {"aliases": ["Estonia"], "capital": ("Tallinn", ["Tallinn"]), "currency": ("euro", ["euro", "Euro"]), "language": ("Estonian", ["Estonian"]), "continent": ("Europe", ["Europe"])},
    "Czech Republic": {"aliases": ["Czech Republic", "the Czech Republic", "Czechia"], "capital": ("Prague", ["Prague"]), "currency": ("koruna", ["koruna", "Czech koruna", "crown"]), "language": ("Czech", ["Czech"]), "continent": ("Europe", ["Europe"])},
    "Portugal": {"aliases": ["Portugal"], "capital": ("Lisbon", ["Lisbon"]), "currency": ("euro", ["euro", "Euro"]), "language": ("Portuguese", ["Portuguese"]), "continent": ("Europe", ["Europe"])},
    "Belgium": {"aliases": ["Belgium"], "capital": ("Brussels", ["Brussels"]), "currency": ("euro", ["euro", "Euro"]), "language": ("Dutch", ["Dutch", "French", "Flemish"]), "continent": ("Europe", ["Europe"])},
    "Denmark": {"aliases": ["Denmark"], "capital": ("Copenhagen", ["Copenhagen"]), "currency": ("krone", ["krone", "Danish krone"]), "language": ("Danish", ["Danish"]), "continent": ("Europe", ["Europe"])},
    "Finland": {"aliases": ["Finland"], "capital": ("Helsinki", ["Helsinki"]), "currency": ("euro", ["euro", "Euro"]), "language": ("Finnish", ["Finnish"]), "continent": ("Europe", ["Europe"])},
    "Slovenia": {"aliases": ["Slovenia"], "capital": ("Ljubljana", ["Ljubljana"]), "currency": ("euro", ["euro", "Euro"]), "language": ("Slovene", ["Slovene", "Slovenian"]), "continent": ("Europe", ["Europe"])},
    "Croatia": {"aliases": ["Croatia"], "capital": ("Zagreb", ["Zagreb"]), "currency": ("euro", ["euro", "Euro", "kuna"]), "language": ("Croatian", ["Croatian"]), "continent": ("Europe", ["Europe"])},
    "Romania": {"aliases": ["Romania"], "capital": ("Bucharest", ["Bucharest"]), "currency": ("leu", ["leu", "Romanian leu"]), "language": ("Romanian", ["Romanian"]), "continent": ("Europe", ["Europe"])},
    "Hungary": {"aliases": ["Hungary"], "capital": ("Budapest", ["Budapest"]), "currency": ("forint", ["forint", "Hungarian forint"]), "language": ("Hungarian", ["Hungarian"]), "continent": ("Europe", ["Europe"])},
    "Poland": {"aliases": ["Poland"], "capital": ("Warsaw", ["Warsaw"]), "currency": ("zloty", ["zloty", "Polish zloty", "złoty"]), "language": ("Polish", ["Polish"]), "continent": ("Europe", ["Europe"])},
    "Sweden": {"aliases": ["Sweden"], "capital": ("Stockholm", ["Stockholm"]), "currency": ("krona", ["krona", "Swedish krona"]), "language": ("Swedish", ["Swedish"]), "continent": ("Europe", ["Europe"])},
    "Norway": {"aliases": ["Norway"], "capital": ("Oslo", ["Oslo"]), "currency": ("krone", ["krone", "Norwegian krone"]), "language": ("Norwegian", ["Norwegian"]), "continent": ("Europe", ["Europe"])},
    "Ireland": {"aliases": ["Ireland"], "capital": ("Dublin", ["Dublin"]), "currency": ("euro", ["euro", "Euro"]), "language": ("English", ["English", "Irish"]), "continent": ("Europe", ["Europe"])},
    "New Zealand": {"aliases": ["New Zealand"], "capital": ("Wellington", ["Wellington"]), "currency": ("dollar", ["dollar", "New Zealand dollar"]), "language": ("English", ["English", "Maori"]), "continent": ("Oceania", ["Oceania", "Australia"])},
    "Australia": {"aliases": ["Australia"], "capital": ("Canberra", ["Canberra"]), "currency": ("dollar", ["dollar", "Australian dollar"]), "language": ("English", ["English"]), "continent": ("Oceania", ["Oceania", "Australia"])},
    "South Korea": {"aliases": ["South Korea", "Korea"], "capital": ("Seoul", ["Seoul"]), "currency": ("won", ["won", "South Korean won", "Korean won"]), "language": ("Korean", ["Korean"]), "continent": ("Asia", ["Asia"])},
    "Thailand": {"aliases": ["Thailand"], "capital": ("Bangkok", ["Bangkok"]), "currency": ("baht", ["baht", "Thai baht"]), "language": ("Thai", ["Thai"]), "continent": ("Asia", ["Asia"])},
    "Malaysia": {"aliases": ["Malaysia"], "capital": ("Kuala Lumpur", ["Kuala Lumpur"]), "currency": ("ringgit", ["ringgit", "Malaysian ringgit"]), "language": ("Malay", ["Malay", "Malaysian"]), "continent": ("Asia", ["Asia"])},
    "Indonesia": {"aliases": ["Indonesia"], "capital": ("Jakarta", ["Jakarta"]), "currency": ("rupiah", ["rupiah", "Indonesian rupiah"]), "language": ("Indonesian", ["Indonesian"]), "continent": ("Asia", ["Asia"])},
    "Philippines": {"aliases": ["Philippines", "the Philippines"], "capital": ("Manila", ["Manila"]), "currency": ("peso", ["peso", "Philippine peso"]), "language": ("Filipino", ["Filipino", "Tagalog", "English"]), "continent": ("Asia", ["Asia"])},
    "Chile": {"aliases": ["Chile"], "capital": ("Santiago", ["Santiago"]), "currency": ("peso", ["peso", "Chilean peso"]), "language": ("Spanish", ["Spanish"]), "continent": ("South America", ["South America"])},
    "Peru": {"aliases": ["Peru"], "capital": ("Lima", ["Lima"]), "currency": ("sol", ["sol", "Peruvian sol"]), "language": ("Spanish", ["Spanish"]), "continent": ("South America", ["South America"])},
    "Colombia": {"aliases": ["Colombia"], "capital": ("Bogota", ["Bogota", "Bogotá"]), "currency": ("peso", ["peso", "Colombian peso"]), "language": ("Spanish", ["Spanish"]), "continent": ("South America", ["South America"])},
    "Argentina": {"aliases": ["Argentina"], "capital": ("Buenos Aires", ["Buenos Aires"]), "currency": ("peso", ["peso", "Argentine peso"]), "language": ("Spanish", ["Spanish"]), "continent": ("South America", ["South America"])},
    "Mexico": {"aliases": ["Mexico"], "capital": ("Mexico City", ["Mexico City"]), "currency": ("peso", ["peso", "Mexican peso"]), "language": ("Spanish", ["Spanish"]), "continent": ("North America", ["North America"])},
    "Canada": {"aliases": ["Canada"], "capital": ("Ottawa", ["Ottawa"]), "currency": ("dollar", ["dollar", "Canadian dollar"]), "language": ("English", ["English", "French"]), "continent": ("North America", ["North America"])},
    "Morocco": {"aliases": ["Morocco"], "capital": ("Rabat", ["Rabat"]), "currency": ("dirham", ["dirham", "Moroccan dirham"]), "language": ("Arabic", ["Arabic"]), "continent": ("Africa", ["Africa"])},
    "Egypt": {"aliases": ["Egypt"], "capital": ("Cairo", ["Cairo"]), "currency": ("pound", ["pound", "Egyptian pound"]), "language": ("Arabic", ["Arabic"]), "continent": ("Africa", ["Africa"])},
    "Kenya": {"aliases": ["Kenya"], "capital": ("Nairobi", ["Nairobi"]), "currency": ("shilling", ["shilling", "Kenyan shilling"]), "language": ("Swahili", ["Swahili", "English"]), "continent": ("Africa", ["Africa"])},
    "Ghana": {"aliases": ["Ghana"], "capital": ("Accra", ["Accra"]), "currency": ("cedi", ["cedi", "Ghanaian cedi"]), "language": ("English", ["English"]), "continent": ("Africa", ["Africa"])},
    "South Africa": {"aliases": ["South Africa"], "capital": ("Pretoria", ["Pretoria", "Cape Town"]), "currency": ("rand", ["rand", "South African rand"]), "language": ("English", ["English", "Zulu", "Afrikaans"]), "continent": ("Africa", ["Africa"])},
    "Iran": {"aliases": ["Iran"], "capital": ("Tehran", ["Tehran"]), "currency": ("rial", ["rial", "Iranian rial"]), "language": ("Persian", ["Persian", "Farsi"]), "continent": ("Asia", ["Asia"])},
    "Uzbekistan": {"aliases": ["Uzbekistan"], "capital": ("Tashkent", ["Tashkent"]), "currency": ("som", ["som", "Uzbekistani som", "sum"]), "language": ("Uzbek", ["Uzbek"]), "continent": ("Asia", ["Asia"])},
    "Kazakhstan": {"aliases": ["Kazakhstan"], "capital": ("Astana", ["Astana", "Nur-Sultan"]), "currency": ("tenge", ["tenge", "Kazakhstani tenge"]), "language": ("Kazakh", ["Kazakh", "Russian"]), "continent": ("Asia", ["Asia"])},
    "Azerbaijan": {"aliases": ["Azerbaijan"], "capital": ("Baku", ["Baku"]), "currency": ("manat", ["manat", "Azerbaijani manat"]), "language": ("Azerbaijani", ["Azerbaijani", "Azeri"]), "continent": ("Asia", ["Asia", "Europe"])},
    "Georgia": {"aliases": ["Georgia"], "capital": ("Tbilisi", ["Tbilisi"]), "currency": ("lari", ["lari", "Georgian lari"]), "language": ("Georgian", ["Georgian"]), "continent": ("Europe", ["Europe", "Asia"])},
    "Armenia": {"aliases": ["Armenia"], "capital": ("Yerevan", ["Yerevan"]), "currency": ("dram", ["dram", "Armenian dram"]), "language": ("Armenian", ["Armenian"]), "continent": ("Asia", ["Asia", "Europe"])},
    "Greece": {"aliases": ["Greece"], "capital": ("Athens", ["Athens"]), "currency": ("euro", ["euro", "Euro"]), "language": ("Greek", ["Greek"]), "continent": ("Europe", ["Europe"])},
    "Turkey": {"aliases": ["Turkey", "Türkiye"], "capital": ("Ankara", ["Ankara"]), "currency": ("lira", ["lira", "Turkish lira"]), "language": ("Turkish", ["Turkish"]), "continent": ("Asia", ["Asia", "Europe"])},
    "Israel": {"aliases": ["Israel"], "capital": ("Jerusalem", ["Jerusalem", "Tel Aviv"]), "currency": ("shekel", ["shekel", "Israeli shekel", "new shekel"]), "language": ("Hebrew", ["Hebrew"]), "continent": ("Asia", ["Asia"])},
    "Netherlands": {"aliases": ["Netherlands", "the Netherlands", "Holland"], "capital": ("Amsterdam", ["Amsterdam"]), "currency": ("euro", ["euro", "Euro"]), "language": ("Dutch", ["Dutch"]), "continent": ("Europe", ["Europe"])},
    "Switzerland": {"aliases": ["Switzerland"], "capital": ("Bern", ["Bern", "Berne"]), "currency": ("franc", ["franc", "Swiss franc"]), "language": ("German", ["German", "French", "Italian"]), "continent": ("Europe", ["Europe"])},
    "Spain": {"aliases": ["Spain"], "capital": ("Madrid", ["Madrid"]), "currency": ("euro", ["euro", "Euro"]), "language": ("Spanish", ["Spanish"]), "continent": ("Europe", ["Europe"])},
    "Slovakia": {"aliases": ["Slovakia"], "capital": ("Bratislava", ["Bratislava"]), "currency": ("euro", ["euro", "Euro"]), "language": ("Slovak", ["Slovak"]), "continent": ("Europe", ["Europe"])},
    "Jordan": {"aliases": ["Jordan"], "capital": ("Amman", ["Amman"]), "currency": ("dinar", ["dinar", "Jordanian dinar"]), "language": ("Arabic", ["Arabic"]), "continent": ("Asia", ["Asia"])},
    "Cambodia": {"aliases": ["Cambodia"], "capital": ("Phnom Penh", ["Phnom Penh"]), "currency": ("riel", ["riel", "Cambodian riel"]), "language": ("Khmer", ["Khmer", "Cambodian"]), "continent": ("Asia", ["Asia"])},
    "Guatemala": {"aliases": ["Guatemala"], "capital": ("Guatemala City", ["Guatemala City"]), "currency": ("quetzal", ["quetzal", "Guatemalan quetzal"]), "language": ("Spanish", ["Spanish"]), "continent": ("North America", ["North America", "Central America"])},
    "Myanmar": {"aliases": ["Myanmar", "Burma"], "capital": ("Naypyidaw", ["Naypyidaw", "Yangon"]), "currency": ("kyat", ["kyat", "Burmese kyat"]), "language": ("Burmese", ["Burmese"]), "continent": ("Asia", ["Asia"])},
    "Sri Lanka": {"aliases": ["Sri Lanka"], "capital": ("Colombo", ["Colombo", "Sri Jayawardenepura Kotte"]), "currency": ("rupee", ["rupee", "Sri Lankan rupee"]), "language": ("Sinhala", ["Sinhala", "Tamil"]), "continent": ("Asia", ["Asia"])},
    "Mali": {"aliases": ["Mali"], "capital": ("Bamako", ["Bamako"]), "currency": ("franc", ["franc", "CFA franc"]), "language": ("French", ["French"]), "continent": ("Africa", ["Africa"])},
    "Singapore": {"aliases": ["Singapore"], "capital": ("Singapore", ["Singapore"]), "currency": ("dollar", ["dollar", "Singapore dollar"]), "language": ("English", ["English", "Malay", "Mandarin"]), "continent": ("Asia", ["Asia"])},
    "Vietnam": {"aliases": ["Vietnam"], "capital": ("Hanoi", ["Hanoi"]), "currency": ("dong", ["dong", "Vietnamese dong"]), "language": ("Vietnamese", ["Vietnamese"]), "continent": ("Asia", ["Asia"])},
    "Ukraine": {"aliases": ["Ukraine"], "capital": ("Kyiv", ["Kyiv", "Kiev"]), "currency": ("hryvnia", ["hryvnia", "Ukrainian hryvnia"]), "language": ("Ukrainian", ["Ukrainian"]), "continent": ("Europe", ["Europe"])},
    "Venezuela": {"aliases": ["Venezuela"], "capital": ("Caracas", ["Caracas"]), "currency": ("bolivar", ["bolivar", "bolívar"]), "language": ("Spanish", ["Spanish"]), "continent": ("South America", ["South America"])},
}

# 32 iconic landmarks across 10 countries (>=3 per country for same-country neighbor checks & LOO class means!)
# Format: (name, shared_prefix, country, struct_aliases, creator_or_fact_prompt_suffix, creator_aliases)
LANDMARKS_T0: List[dict] = [
    # France (4)
    {"name": "Eiffel Tower", "prefix": "The Eiffel Tower", "country": "France",
     "struct_aliases": ["tower", "iron", "landmark", "monument", "lattice", "structure"],
     "prot2_suffix": " was designed by", "prot2_aliases": ["Gustave", "Eiffel"]},
    {"name": "Louvre", "prefix": "The Louvre", "country": "France",
     "struct_aliases": ["museum", "art", "palace", "gallery", "landmark"],
     "prot2_suffix": " houses the famous painting known as the", "prot2_aliases": ["Mona", "Lisa"]},
    {"name": "Notre-Dame", "prefix": "Notre-Dame", "country": "France",
     "struct_aliases": ["cathedral", "church", "Gothic", "landmark", "monument"],
     "prot2_suffix": " is an iconic Gothic", "prot2_aliases": ["cathedral", "church"]},
    {"name": "Arc de Triomphe", "prefix": "The Arc de Triomphe", "country": "France",
     "struct_aliases": ["monument", "arch", "landmark", "memorial", "triumphal"],
     "prot2_suffix": " was commissioned by", "prot2_aliases": ["Napoleon", "Bonaparte"]},
    # Japan (4)
    {"name": "Tokyo Tower", "prefix": "Tokyo Tower", "country": "Japan",
     "struct_aliases": ["tower", "communications", "observation", "landmark", "broadcasting"],
     "prot2_suffix": " was inspired by the", "prot2_aliases": ["Eiffel", "Tower"]},
    {"name": "Mount Fuji", "prefix": "Mount Fuji", "country": "Japan",
     "struct_aliases": ["mountain", "volcano", "peak", "stratovolcano", "landmark"],
     "prot2_suffix": " is an active", "prot2_aliases": ["volcano", "stratovolcano", "mountain"]},
    {"name": "Senso-ji", "prefix": "Senso-ji", "country": "Japan",
     "struct_aliases": ["temple", "Buddhist", "shrine", "landmark"],
     "prot2_suffix": " is an ancient Buddhist", "prot2_aliases": ["temple", "shrine"]},
    {"name": "Osaka Castle", "prefix": "Osaka Castle", "country": "Japan",
     "struct_aliases": ["castle", "fortress", "landmark", "palace", "historic"],
     "prot2_suffix": " is a historic", "prot2_aliases": ["castle", "fortress", "Japanese"]},
    # United States (4)
    {"name": "Statue of Liberty", "prefix": "The Statue of Liberty", "country": "United States",
     "struct_aliases": ["statue", "monument", "sculpture", "landmark", "copper"],
     "prot2_suffix": " was a gift from", "prot2_aliases": ["France", "the French"]},
    {"name": "Empire State Building", "prefix": "The Empire State Building", "country": "United States",
     "struct_aliases": ["skyscraper", "building", "tower", "landmark", "Art Deco"],
     "prot2_suffix": " is a famous Art Deco", "prot2_aliases": ["skyscraper", "building", "tower"]},
    {"name": "Golden Gate Bridge", "prefix": "The Golden Gate Bridge", "country": "United States",
     "struct_aliases": ["bridge", "suspension", "landmark", "structure"],
     "prot2_suffix": " is an iconic suspension", "prot2_aliases": ["bridge"]},
    {"name": "Mount Rushmore", "prefix": "Mount Rushmore", "country": "United States",
     "struct_aliases": ["monument", "memorial", "sculpture", "mountain", "landmark", "carving"],
     "prot2_suffix": " features the carved faces of four", "prot2_aliases": ["presidents", "U.S.", "US", "American"]},
    # Italy (4)
    {"name": "Colosseum", "prefix": "The Colosseum", "country": "Italy",
     "struct_aliases": ["amphitheater", "amphitheatre", "arena", "stadium", "landmark", "monument", "Roman"],
     "prot2_suffix": " was used in ancient times for", "prot2_aliases": ["gladiator", "gladiatorial", "contests", "games", "battles", "fights"]},
    {"name": "Leaning Tower of Pisa", "prefix": "The Leaning Tower of Pisa", "country": "Italy",
     "struct_aliases": ["tower", "bell", "campanile", "landmark", "monument"],
     "prot2_suffix": " is famous for its unintended", "prot2_aliases": ["tilt", "lean", "leaning", "angle"]},
    {"name": "Trevi Fountain", "prefix": "The Trevi Fountain", "country": "Italy",
     "struct_aliases": ["fountain", "Baroque", "landmark", "monument", "water"],
     "prot2_suffix": " is a world-famous Baroque", "prot2_aliases": ["fountain"]},
    {"name": "St. Peter's Basilica", "prefix": "St. Peter's Basilica", "country": "Italy",
     "struct_aliases": ["church", "basilica", "cathedral", "Renaissance", "landmark"],
     "prot2_suffix": " is the largest", "prot2_aliases": ["church", "basilica", "Christian", "Catholic"]},
    # United Kingdom (3)
    {"name": "Big Ben", "prefix": "Big Ben", "country": "United Kingdom",
     "struct_aliases": ["clock", "tower", "bell", "landmark", "monument"],
     "prot2_suffix": " is the nickname of the great", "prot2_aliases": ["bell", "clock", "tower"]},
    {"name": "Tower of London", "prefix": "The Tower of London", "country": "United Kingdom",
     "struct_aliases": ["castle", "fortress", "palace", "prison", "tower", "landmark"],
     "prot2_suffix": " is home to the British", "prot2_aliases": ["Crown", "crown", "Jewels", "monarchy"]},
    {"name": "Stonehenge", "prefix": "Stonehenge", "country": "United Kingdom",
     "struct_aliases": ["monument", "prehistoric", "stone", "circle", "megalithic", "landmark"],
     "prot2_suffix": " is a prehistoric ring of standing", "prot2_aliases": ["stones", "stone", "megaliths"]},
    # Germany (3)
    {"name": "Brandenburg Gate", "prefix": "The Brandenburg Gate", "country": "Germany",
     "struct_aliases": ["monument", "gate", "arch", "landmark", "neoclassical"],
     "prot2_suffix": " is an eighteenth-century neoclassical", "prot2_aliases": ["monument", "gate", "arch"]},
    {"name": "Neuschwanstein Castle", "prefix": "Neuschwanstein Castle", "country": "Germany",
     "struct_aliases": ["castle", "palace", "landmark", "fortress"],
     "prot2_suffix": " was commissioned by King", "prot2_aliases": ["Ludwig", "Louis"]},
    {"name": "Cologne Cathedral", "prefix": "Cologne Cathedral", "country": "Germany",
     "struct_aliases": ["cathedral", "church", "Gothic", "landmark"],
     "prot2_suffix": " is a renowned Catholic", "prot2_aliases": ["cathedral", "church"]},
    # India (3)
    {"name": "Taj Mahal", "prefix": "The Taj Mahal", "country": "India",
     "struct_aliases": ["mausoleum", "monument", "tomb", "palace", "marble", "landmark"],
     "prot2_suffix": " was built by Emperor", "prot2_aliases": ["Shah", "Jahan"]},
    {"name": "Red Fort", "prefix": "The Red Fort", "country": "India",
     "struct_aliases": ["fort", "fortress", "palace", "monument", "landmark"],
     "prot2_suffix": " is a historic Mughal", "prot2_aliases": ["fort", "fortress", "palace"]},
    {"name": "Gateway of India", "prefix": "The Gateway of India", "country": "India",
     "struct_aliases": ["monument", "arch", "landmark", "gateway"],
     "prot2_suffix": " is a historic arch", "prot2_aliases": ["monument", "built", "located"]},
    # Brazil (3)
    {"name": "Christ the Redeemer", "prefix": "Christ the Redeemer", "country": "Brazil",
     "struct_aliases": ["statue", "monument", "sculpture", "landmark", "Art Deco"],
     "prot2_suffix": " is a giant Art Deco", "prot2_aliases": ["statue", "sculpture", "monument"]},
    {"name": "Sugarloaf Mountain", "prefix": "Sugarloaf Mountain", "country": "Brazil",
     "struct_aliases": ["mountain", "peak", "hill", "landmark"],
     "prot2_suffix": " is a famous granite", "prot2_aliases": ["peak", "mountain", "hill", "rock"]},
    {"name": "Iguazu Falls", "prefix": "Iguazu Falls", "country": "Brazil",
     "struct_aliases": ["waterfall", "waterfalls", "falls", "landmark", "natural"],
     "prot2_suffix": " is one of the largest", "prot2_aliases": ["waterfall", "waterfalls", "falls"]},
    # Russia (2)
    {"name": "Kremlin", "prefix": "The Kremlin", "country": "Russia",
     "struct_aliases": ["fortress", "complex", "palace", "citadel", "landmark", "government"],
     "prot2_suffix": " serves as the official residence of the", "prot2_aliases": ["President", "president", "Russian"]},
    {"name": "Hermitage Museum", "prefix": "The Hermitage Museum", "country": "Russia",
     "struct_aliases": ["museum", "art", "palace", "gallery", "landmark"],
     "prot2_suffix": " is one of the world's largest", "prot2_aliases": ["art", "museums", "museum"]},
    # China (2)
    {"name": "Great Wall of China", "prefix": "The Great Wall of China", "country": "China",
     "struct_aliases": ["wall", "fortification", "fortifications", "monument", "landmark", "series"],
     "prot2_suffix": " was built to protect against", "prot2_aliases": ["invasions", "nomadic", "invaders", "Mongol", "northern", "raids"]},
    {"name": "Forbidden City", "prefix": "The Forbidden City", "country": "China",
     "struct_aliases": ["palace", "complex", "imperial", "monument", "landmark"],
     "prot2_suffix": " served as the imperial", "prot2_aliases": ["palace", "residence", "home", "court"]},
]

# Fallback pool of 24 famous people (used ONLY if < 12 landmarks pass T0 competence on SmolLM2!)
PEOPLE_FALLBACK_T0: List[dict] = [
    {"name": "Marie Curie", "prefix": "Marie Curie", "country": "Poland", "struct_aliases": ["physicist", "chemist", "scientist"], "prot2_suffix": " won the Nobel Prize in", "prot2_aliases": ["Physics", "Chemistry", "physics", "chemistry"]},
    {"name": "Albert Einstein", "prefix": "Albert Einstein", "country": "Germany", "struct_aliases": ["physicist", "scientist", "theoretical"], "prot2_suffix": " developed the theory of", "prot2_aliases": ["relativity", "Relativity"]},
    {"name": "Isaac Newton", "prefix": "Isaac Newton", "country": "United Kingdom", "struct_aliases": ["physicist", "mathematician", "scientist"], "prot2_suffix": " formulated the laws of", "prot2_aliases": ["motion", "gravity", "gravitation"]},
    {"name": "Galileo Galilei", "prefix": "Galileo Galilei", "country": "Italy", "struct_aliases": ["astronomer", "physicist", "scientist"], "prot2_suffix": " is known as the father of", "prot2_aliases": ["modern", "observational", "astronomy", "science"]},
    {"name": "Louis Pasteur", "prefix": "Louis Pasteur", "country": "France", "struct_aliases": ["chemist", "microbiologist", "scientist"], "prot2_suffix": " invented the process of", "prot2_aliases": ["pasteurization", "Pasteurization"]},
    {"name": "Charles Darwin", "prefix": "Charles Darwin", "country": "United Kingdom", "struct_aliases": ["naturalist", "biologist", "scientist"], "prot2_suffix": " proposed the theory of", "prot2_aliases": ["evolution", "natural"]},
    {"name": "Wolfgang Mozart", "prefix": "Wolfgang Mozart", "country": "Austria", "struct_aliases": ["composer", "musician", "pianist"], "prot2_suffix": " was a prolific classical", "prot2_aliases": ["composer", "musician"]},
    {"name": "Ludwig van Beethoven", "prefix": "Ludwig van Beethoven", "country": "Germany", "struct_aliases": ["composer", "pianist", "musician"], "prot2_suffix": " composed the Ninth", "prot2_aliases": ["Symphony", "symphony"]},
    {"name": "Leonardo da Vinci", "prefix": "Leonardo da Vinci", "country": "Italy", "struct_aliases": ["artist", "painter", "polymath", "inventor"], "prot2_suffix": " painted the", "prot2_aliases": ["Mona", "Last"]},
    {"name": "Pablo Picasso", "prefix": "Pablo Picasso", "country": "Spain", "struct_aliases": ["painter", "artist", "sculptor"], "prot2_suffix": " co-founded the movement known as", "prot2_aliases": ["Cubism", "cubism"]},
    {"name": "Leo Tolstoy", "prefix": "Leo Tolstoy", "country": "Russia", "struct_aliases": ["writer", "author", "novelist"], "prot2_suffix": " wrote the novel War and", "prot2_aliases": ["Peace"]},
    {"name": "Fyodor Dostoevsky", "prefix": "Fyodor Dostoevsky", "country": "Russia", "struct_aliases": ["writer", "author", "novelist"], "prot2_suffix": " wrote Crime and", "prot2_aliases": ["Punishment"]},
    {"name": "William Shakespeare", "prefix": "William Shakespeare", "country": "United Kingdom", "struct_aliases": ["playwright", "poet", "writer", "author"], "prot2_suffix": " wrote the play", "prot2_aliases": ["Hamlet", "Romeo", "Macbeth", "Othello"]},
    {"name": "Victor Hugo", "prefix": "Victor Hugo", "country": "France", "struct_aliases": ["writer", "author", "poet", "novelist"], "prot2_suffix": " wrote Les", "prot2_aliases": ["Miserables", "Misérables"]},
    {"name": "Akira Kurosawa", "prefix": "Akira Kurosawa", "country": "Japan", "struct_aliases": ["director", "filmmaker", "film"], "prot2_suffix": " directed the film Seven", "prot2_aliases": ["Samurai"]},
    {"name": "Hayao Miyazaki", "prefix": "Hayao Miyazaki", "country": "Japan", "struct_aliases": ["animator", "director", "filmmaker"], "prot2_suffix": " co-founded Studio", "prot2_aliases": ["Ghibli"]},
    {"name": "Pele", "prefix": "Pele", "country": "Brazil", "struct_aliases": ["footballer", "soccer", "athlete", "player"], "prot2_suffix": " was a legendary", "prot2_aliases": ["Brazilian", "football", "soccer"]},
    {"name": "Rabindranath Tagore", "prefix": "Rabindranath Tagore", "country": "India", "struct_aliases": ["poet", "writer", "author", "philosopher"], "prot2_suffix": " won the Nobel Prize in", "prot2_aliases": ["Literature", "literature"]},
    {"name": "Confucius", "prefix": "Confucius", "country": "China", "struct_aliases": ["philosopher", "teacher", "thinker"], "prot2_suffix": " was an ancient Chinese", "prot2_aliases": ["philosopher", "teacher", "thinker"]},
    {"name": "Thomas Edison", "prefix": "Thomas Edison", "country": "United States", "struct_aliases": ["inventor", "businessman", "scientist"], "prot2_suffix": " invented the practical electric light", "prot2_aliases": ["bulb"]},
]

# T3: ~145 medium/long-tail real-world entities (cities, universities, landmarks, rivers)
# Format: (entity_name, entity_kind, country)
T3_ENTITIES: List[Tuple[str, str, str]] = [
    # Medium/long-tail European cities
    ("Graz", "city", "Austria"), ("Innsbruck", "city", "Austria"), ("Linz", "city", "Austria"),
    ("Plovdiv", "city", "Bulgaria"), ("Varna", "city", "Bulgaria"),
    ("Tartu", "city", "Estonia"),
    ("Brno", "city", "Czech Republic"), ("Ostrava", "city", "Czech Republic"),
    ("Coimbra", "city", "Portugal"), ("Faro", "city", "Portugal"), ("Braga", "city", "Portugal"),
    ("Ghent", "city", "Belgium"), ("Antwerp", "city", "Belgium"), ("Bruges", "city", "Belgium"),
    ("Aarhus", "city", "Denmark"), ("Odense", "city", "Denmark"),
    ("Tampere", "city", "Finland"), ("Oulu", "city", "Finland"), ("Turku", "city", "Finland"),
    ("Maribor", "city", "Slovenia"),
    ("Split", "city", "Croatia"), ("Dubrovnik", "city", "Croatia"), ("Rijeka", "city", "Croatia"),
    ("Cluj-Napoca", "city", "Romania"), ("Timisoara", "city", "Romania"), ("Brasov", "city", "Romania"),
    ("Szeged", "city", "Hungary"), ("Debrecen", "city", "Hungary"), ("Pecs", "city", "Hungary"),
    ("Gdansk", "city", "Poland"), ("Wroclaw", "city", "Poland"), ("Poznan", "city", "Poland"), ("Lodz", "city", "Poland"),
    ("Gothenburg", "city", "Sweden"), ("Malmo", "city", "Sweden"), ("Uppsala", "city", "Sweden"),
    ("Bergen", "city", "Norway"), ("Trondheim", "city", "Norway"), ("Stavanger", "city", "Norway"),
    ("Cork", "city", "Ireland"), ("Galway", "city", "Ireland"), ("Limerick", "city", "Ireland"),
    ("Kosice", "city", "Slovakia"),
    ("Thessaloniki", "city", "Greece"), ("Patras", "city", "Greece"), ("Heraklion", "city", "Greece"),
    ("Utrecht", "city", "Netherlands"), ("Eindhoven", "city", "Netherlands"), ("Groningen", "city", "Netherlands"),
    ("Basel", "city", "Switzerland"), ("Lausanne", "city", "Switzerland"), ("Lucerne", "city", "Switzerland"),
    ("Bilbao", "city", "Spain"), ("Malaga", "city", "Spain"), ("Zaragoza", "city", "Spain"),
    ("Bologna", "city", "Italy"), ("Verona", "city", "Italy"), ("Palermo", "city", "Italy"), ("Genoa", "city", "Italy"),
    ("Toulouse", "city", "France"), ("Nantes", "city", "France"), ("Lille", "city", "France"), ("Strasbourg", "city", "France"),
    ("Leipzig", "city", "Germany"), ("Bremen", "city", "Germany"), ("Nuremberg", "city", "Germany"), ("Dresden", "city", "Germany"),
    ("Sheffield", "city", "United Kingdom"), ("Bristol", "city", "United Kingdom"), ("Cardiff", "city", "United Kingdom"),
    # Medium/long-tail global cities (Asia, Americas, Africa, Oceania)
    ("Dunedin", "city", "New Zealand"), ("Christchurch", "city", "New Zealand"),
    ("Hobart", "city", "Australia"), ("Adelaide", "city", "Australia"),
    ("Sendai", "city", "Japan"), ("Kanazawa", "city", "Japan"), ("Nagasaki", "city", "Japan"), ("Fukuoka", "city", "Japan"),
    ("Daegu", "city", "South Korea"), ("Incheon", "city", "South Korea"), ("Gwangju", "city", "South Korea"),
    ("Chiang Mai", "city", "Thailand"), ("Phuket", "city", "Thailand"),
    ("Penang", "city", "Malaysia"), ("Johor Bahru", "city", "Malaysia"),
    ("Surabaya", "city", "Indonesia"), ("Bandung", "city", "Indonesia"), ("Yogyakarta", "city", "Indonesia"),
    ("Cebu", "city", "Philippines"), ("Davao", "city", "Philippines"),
    ("Valparaiso", "city", "Chile"), ("Concepcion", "city", "Chile"),
    ("Arequipa", "city", "Peru"), ("Cusco", "city", "Peru"),
    ("Medellin", "city", "Colombia"), ("Cartagena", "city", "Colombia"), ("Cali", "city", "Colombia"),
    ("Cordoba", "city", "Argentina"), ("Rosario", "city", "Argentina"), ("Mendoza", "city", "Argentina"),
    ("Curitiba", "city", "Brazil"), ("Recife", "city", "Brazil"), ("Salvador", "city", "Brazil"),
    ("Guadalajara", "city", "Mexico"), ("Monterrey", "city", "Mexico"), ("Puebla", "city", "Mexico"),
    ("Halifax", "city", "Canada"), ("Winnipeg", "city", "Canada"), ("Saskatoon", "city", "Canada"),
    ("Fez", "city", "Morocco"), ("Marrakech", "city", "Morocco"), ("Tangier", "city", "Morocco"),
    ("Luxor", "city", "Egypt"), ("Aswan", "city", "Egypt"),
    ("Mombasa", "city", "Kenya"), ("Kisumu", "city", "Kenya"),
    ("Kumasi", "city", "Ghana"),
    ("Durban", "city", "South Africa"), ("Bloemfontein", "city", "South Africa"),
    ("Isfahan", "city", "Iran"), ("Shiraz", "city", "Iran"), ("Tabriz", "city", "Iran"),
    ("Samarkand", "city", "Uzbekistan"), ("Bukhara", "city", "Uzbekistan"),
    ("Almaty", "city", "Kazakhstan"),
    ("Izmir", "city", "Turkey"), ("Bursa", "city", "Turkey"), ("Antalya", "city", "Turkey"),
    ("Haifa", "city", "Israel"), ("Eilat", "city", "Israel"),
    ("Hue", "city", "Vietnam"), ("Da Nang", "city", "Vietnam"),
    # Medium/long-tail universities
    ("Uppsala University", "university", "Sweden"),
    ("Leiden University", "university", "Netherlands"),
    ("University of Salamanca", "university", "Spain"),
    ("University of Coimbra", "university", "Portugal"),
    ("Charles University", "university", "Czech Republic"),
    ("Jagiellonian University", "university", "Poland"),
    ("University of Tartu", "university", "Estonia"),
    ("University of Otago", "university", "New Zealand"),
    ("Tohoku University", "university", "Japan"),
    ("Waseda University", "university", "Japan"),
    ("KAIST", "university", "South Korea"),
    ("KU Leuven", "university", "Belgium"),
    ("Karolinska Institute", "university", "Sweden"),
    ("Trinity College Dublin", "university", "Ireland"),
    ("University of St Andrews", "university", "United Kingdom"),
    # Medium/long-tail landmarks & rivers
    ("Wawel Castle", "landmark", "Poland"),
    ("Bran Castle", "landmark", "Romania"),
    ("Blarney Castle", "landmark", "Ireland"),
    ("Himeji Castle", "landmark", "Japan"),
    ("Gyeongbokgung", "landmark", "South Korea"),
    ("Borobudur", "landmark", "Indonesia"),
    ("Bagan", "landmark", "Myanmar"),
    ("Sigiriya", "landmark", "Sri Lanka"),
    ("Tikal", "landmark", "Guatemala"),
    ("Persepolis", "landmark", "Iran"),
    ("Ephesus", "landmark", "Turkey"),
    ("Petra", "landmark", "Jordan"),
    ("Angkor Wat", "landmark", "Cambodia"),
    ("Alhambra", "landmark", "Spain"),
    ("Vistula River", "river", "Poland"),
    ("Loire River", "river", "France"),
    ("Po River", "river", "Italy"),
    ("Shinano River", "river", "Japan"),
    ("Waikato River", "river", "New Zealand"),
    ("Irrawaddy River", "river", "Myanmar"),
]


# ======================================================================================
# 2. Pure-CPU Float64 Linear Algebra Kernels (Procrustes, CCA/PLS Socket, MVDR, Gate 3)
# ======================================================================================

def _polar_f64(M: np.ndarray) -> np.ndarray:
    try:
        U, _, Vt = sla.svd(M, full_matrices=False, lapack_driver="gesdd", check_finite=False)
    except sla.LinAlgError:
        U, _, Vt = sla.svd(M, full_matrices=False, lapack_driver="gesvd", check_finite=False)
    return U @ Vt


def fit_procrustes_f64(Z_A: np.ndarray, Z_B: np.ndarray) -> dict:
    ZA = Z_A.astype(np.float64)
    ZB = Z_B.astype(np.float64)
    mu_A = ZA.mean(axis=0)
    mu_B = ZB.mean(axis=0)
    Ac = ZA - mu_A
    Bc = ZB - mu_B
    R = _polar_f64(Bc.T @ Ac)
    s = float(np.linalg.norm(Bc, "fro") / max(np.linalg.norm(Ac, "fro"), 1e-12))
    return {"mu_A": mu_A, "mu_B": mu_B, "R": R, "s": s}


def fit_platonic_socket_f64(H_A: np.ndarray, H_B: np.ndarray, shrink: float = 0.05) -> dict:
    """
    Fit shared Platonic coordinate socket (U_A, U_B, r_corr, mu_A, mu_B, W_A_white, W_B_white)
    on post-RMSNorm unlabeled calibration states H_A, H_B [N, D] in CPU float64.
    Provides BOTH orthogonal cross-covariance SVD basis (U_A, U_B) AND empirical per-axis
    cross-family correlation r_i = corr(z_A_i, z_B_i) for Wiener r_i^2 weighting!
    """
    A = H_A.astype(np.float64)
    B = H_B.astype(np.float64)
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    Ac = A - mu_A
    Bc = B - mu_B
    n = len(A)
    # Cross-covariance SVD: (1/N) B_c^T A_c = U_B diag(sv) U_A^T
    # Columns of U_B and U_A are orthonormal and ordered by cross-covariance strength!
    C_BA = (Bc.T @ Ac) / max(n, 1)
    U_B, sv, Vt_A = sla.svd(C_BA, full_matrices=False, check_finite=False)
    U_A = Vt_A.T  # [D, D]
    # Project onto aligned basis and measure exact per-coordinate canonical correlation r_i
    ZA_p = Ac @ U_A  # [N, D]
    ZB_p = Bc @ U_B  # [N, D]
    num = np.sum(ZA_p * ZB_p, axis=0)
    den = np.maximum(np.linalg.norm(ZA_p, axis=0) * np.linalg.norm(ZB_p, axis=0), 1e-12)
    r_corr = np.clip(num / den, 0.0, 1.0)
    # Per-coordinate variance on B in socket coordinates for diagonal/full whitening
    sig_B = (ZB_p.T @ ZB_p) / max(n, 1)
    return {
        "mu_A": mu_A,
        "mu_B": mu_B,
        "U_A": U_A,
        "U_B": U_B,
        "sv": sv,
        "r_corr": r_corr,
        "sig_B": sig_B,
    }


def compute_mvdr_filter(
    target_mean: np.ndarray,
    hard_neg_means: np.ndarray,
    C_bg: np.ndarray,
    shrink_rel: float = 0.05,
) -> np.ndarray:
    """
    Fable's constrained minimum-variance (MVDR / Capon) key filter:
      g = C^{-1} K (K^T C^{-1} K)^{-1} e_1
    where K = [target_mean, hard_neg_means...] has columns (D, M),
    e_1 = [1, 0, ..., 0]^T, enforcing g^T target = 1 and g^T hard_neg_j = 0
    while minimizing background variance g^T C g!
    """
    d = C_bg.shape[0]
    tr = max(float(np.trace(C_bg)) / d, 1e-12)
    C_reg = 0.5 * (C_bg + C_bg.T) + (shrink_rel * tr) * np.eye(d, dtype=np.float64)
    # Build K: column 0 is target_mean, remaining columns are unique hard negatives
    cols = [target_mean.astype(np.float64)]
    if hard_neg_means is not None and len(hard_neg_means) > 0:
        for row in hard_neg_means:
            cols.append(row.astype(np.float64))
    K = np.column_stack(cols)  # [d, m]
    # Solve C_reg X = K -> X = C^{-1} K
    c_fac = sla.cho_factor(C_reg, check_finite=False)
    CinvK = sla.cho_solve(c_fac, K, check_finite=False)  # [d, m]
    G_mat = K.T @ CinvK  # [m, m]
    # Tiny ridge on G_mat for numerical stability when hard negatives are collinear
    g_tr = max(float(np.trace(G_mat)) / max(G_mat.shape[0], 1), 1e-12)
    G_reg = G_mat + (1e-4 * g_tr) * np.eye(G_mat.shape[0], dtype=np.float64)
    e1 = np.zeros(G_mat.shape[0], dtype=np.float64)
    e1[0] = 1.0
    alpha = sla.solve(G_reg, e1, assume_a="pos", check_finite=False)
    g = CinvK @ alpha
    g_norm = np.linalg.norm(g)
    return g / max(g_norm, 1e-12)


def compute_dprime_sq(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Squared discriminability index d'^2 = (mu_pos - mu_neg)^2 / (0.5 * (var_pos + var_neg))."""
    mp = float(np.mean(pos_scores))
    mn = float(np.mean(neg_scores))
    vp = float(np.var(pos_scores))
    vn = float(np.var(neg_scores))
    denom = max(0.5 * (vp + vn), 1e-12)
    return float(((mp - mn) ** 2) / denom)


def fit_implicit_constant_gate3(H_tr: np.ndarray, H_ev: np.ndarray, lam_rel: float = 1e-3) -> dict:
    """
    Neon Gate 3 audit: fit c_B = argmin_c E[(c^T h_tilde - 1)^2] + lam ||c||^2
    on post-RMSNorm states H_tr [N_tr, D], evaluate std(c^T h) and p99|c^T h - 1| on H_ev.
    """
    X = H_tr.astype(np.float64)
    n, d = X.shape
    C = (X.T @ X) / max(n, 1)
    tr = max(float(np.trace(C)) / d, 1e-12)
    C_reg = 0.5 * (C + C.T) + (lam_rel * tr) * np.eye(d, dtype=np.float64)
    rhs = X.mean(axis=0)
    c_B = sla.solve(C_reg, rhs, assume_a="pos", check_finite=False)
    vals_tr = X @ c_B
    vals_ev = H_ev.astype(np.float64) @ c_B
    dev_ev = np.abs(vals_ev - 1.0)
    return {
        "train_mean": round(float(np.mean(vals_tr)), 5),
        "train_std": round(float(np.std(vals_tr)), 5),
        "eval_mean": round(float(np.mean(vals_ev)), 5),
        "eval_std_sigma_c": round(float(np.std(vals_ev)), 5),
        "eval_p95_abs_err": round(float(np.percentile(dev_ev, 95)), 5),
        "eval_p99_abs_err_delta99": round(float(np.percentile(dev_ev, 99)), 5),
        "eval_max_abs_err": round(float(np.max(dev_ev)), 5),
        "c_B_norm": round(float(np.linalg.norm(c_B)), 4),
    }


def assign_deranged_targets(items: List[dict], seed: int = SEED) -> Dict[str, str]:
    """
    Assign each landmark a deranged target country c_new != c_old such that:
      1. c_new differs from c_old in country, capital, currency, AND language!
      2. Landmarks sharing the SAME old country receive DIFFERENT target countries!
    """
    all_countries = sorted({it["country"] for it in items})
    rng = np.random.default_rng(seed)

    def _valid_target(c_old: str, c_new: str) -> bool:
        if c_old == c_new:
            return False
        o = COUNTRIES_INFO[c_old]
        n = COUNTRIES_INFO[c_new]
        if o["capital"][0].lower() == n["capital"][0].lower():
            return False
        if o["currency"][0].lower() == n["currency"][0].lower():
            return False
        if o["language"][0].lower() == n["language"][0].lower():
            return False
        return True

    by_old: Dict[str, List[dict]] = defaultdict(list)
    for it in items:
        by_old[it["country"]].append(it)

    mapping: Dict[str, str] = {}
    for c_old in sorted(by_old.keys()):
        valid_cands = [c for c in all_countries if _valid_target(c_old, c)]
        assert len(valid_cands) >= len(by_old[c_old]), (
            f"Too few disjoint target countries for {c_old}: {len(valid_cands)} < {len(by_old[c_old])}"
        )
        perm = rng.permutation(len(valid_cands))
        for idx, it in enumerate(by_old[c_old]):
            mapping[it["name"]] = valid_cands[int(perm[idx])]
    return mapping


# ======================================================================================
# 3. Model Helpers: Batched Forward, Extraction, Generation, Log-Odds & Candidate Choice
# ======================================================================================

def load_model_f32(model_dir: Path, device: torch.device) -> Tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, dtype=torch.float32
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def extract_multi_layer_states(
    model: Any,
    tok: Any,
    prompts: List[str],
    char_spans: List[Tuple[int, int]],
    layer_indices: List[int],
    device: torch.device,
    capture_mlp_in: bool = False,
    batch_size: int = 32,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Single forward sweep capturing BOTH post-decoder residual stream `resid[l]` AND
    (optionally) post-RMSNorm MLP input `mlp_in[l]` at last subject token of each prompt!
    """
    layers = get_decoder_layers(model)
    n = len(prompts)
    D = int(model.config.hidden_size)
    resid_out = {l: np.zeros((n, D), dtype=np.float32) for l in layer_indices}
    mlp_in_out = {l: np.zeros((n, D), dtype=np.float32) for l in layer_indices} if capture_mlp_in else {}

    # Precompute exact subject token positions
    subj_idxs = []
    seq_lens = []
    for p, sp in zip(prompts, char_spans):
        si, sl = locate_last_subject_token(tok, p, sp)
        subj_idxs.append(si)
        seq_lens.append(sl)

    ctx: Dict[str, Any] = {"pos": None, "resid_buf": {}, "mlp_buf": {}}
    handles = []

    for l in layer_indices:
        def _make_resid_hook(li: int):
            def _hk(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                b_ar = torch.arange(h.shape[0], device=h.device)
                ctx["resid_buf"][li] = h[b_ar, ctx["pos"], :].detach().float().cpu().numpy()
            return _hk
        handles.append(layers[l].register_forward_hook(_make_resid_hook(l)))

        if capture_mlp_in:
            def _make_mlp_pre_hook(li: int):
                def _hk(_mod, inp):
                    h = inp[0]
                    b_ar = torch.arange(h.shape[0], device=h.device)
                    ctx["mlp_buf"][li] = h[b_ar, ctx["pos"], :].detach().float().cpu().numpy()
                return _hk
            handles.append(layers[l].mlp.register_forward_pre_hook(_make_mlp_pre_hook(l)))

    try:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            enc = tok(prompts[start:end], return_tensors="pt", padding=True, add_special_tokens=True)
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)
            t_pad = ids.shape[1]
            pos = [max(0, min(t_pad - 1, (t_pad - seq_lens[i]) + subj_idxs[i])) for i in range(start, end)]
            ctx["pos"] = torch.tensor(pos, dtype=torch.long, device=device)
            ctx["resid_buf"].clear()
            ctx["mlp_buf"].clear()
            with torch.no_grad():
                model(input_ids=ids, attention_mask=am, use_cache=False)
            for l in layer_indices:
                resid_out[l][start:end] = ctx["resid_buf"][l]
                if capture_mlp_in:
                    mlp_in_out[l][start:end] = ctx["mlp_buf"][l]
    finally:
        for h in handles:
            h.remove()

    return {"resid": resid_out, "mlp_in": mlp_in_out}


def extract_all_tokens_mlp_in(
    model: Any,
    tok: Any,
    prompts: List[str],
    layer_idx: int,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract post-RMSNorm MLP-input states for ALL real tokens of `prompts`,
    returning (non_bos_tokens [M1, D], bos_tokens [M2, D]).
    """
    layers = get_decoder_layers(model)
    non_bos_list = []
    bos_list = []
    ctx: Dict[str, Any] = {"buf": None}

    def _hk(_mod, inp):
        ctx["buf"] = inp[0].detach().float().cpu().numpy()

    handle = layers[layer_idx].mlp.register_forward_pre_hook(_hk)
    try:
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            enc = tok(prompts[start:end], return_tensors="pt", padding=True, add_special_tokens=True)
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)
            with torch.no_grad():
                model(input_ids=ids, attention_mask=am, use_cache=False)
            arr = ctx["buf"]  # [B, T, D]
            am_np = am.cpu().numpy()
            for bi in range(arr.shape[0]):
                valid_pos = np.where(am_np[bi] == 1)[0]
                if len(valid_pos) == 0:
                    continue
                bos_list.append(arr[bi, valid_pos[0]])
                if len(valid_pos) > 1:
                    non_bos_list.append(arr[bi, valid_pos[1:]])
    finally:
        handle.remove()

    non_bos = np.vstack(non_bos_list) if non_bos_list else np.zeros((0, int(model.config.hidden_size)), dtype=np.float32)
    bos = np.vstack(bos_list) if bos_list else np.zeros((0, int(model.config.hidden_size)), dtype=np.float32)
    return non_bos, bos


def _first_tok_id(tok: Any, text: str) -> int:
    ids = tok.encode(" " + text, add_special_tokens=False)
    if not ids:
        ids = tok.encode(text, add_special_tokens=False)
    return int(ids[0])


def eval_prompts_patched(
    model: Any,
    tok: Any,
    prompts: List[str],
    char_spans: List[Tuple[int, int]],
    target_aliases_list: List[List[str]],
    old_aliases_list: Optional[List[List[str]]] = None,
    candidate_pool_list: Optional[List[List[str]]] = None,
    layer_idx: Optional[int] = None,
    delta_mat: Optional[np.ndarray] = None,
    replace_mat: Optional[np.ndarray] = None,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 32,
    max_new_tokens: int = 6,
) -> dict:
    """
    Evaluate prompts (unpatched, additive residual delta `delta_mat`, OR residual replace `replace_mat`)
    at `(layer_idx, last_subj)`:
      - Exact greedy top-1 EM (`check_answer_match`, 6 tokens)
      - Candidate-pool first-token choice accuracy (if `candidate_pool_list` given)
      - Mean log P(new_first_tok) and log-odds `log P(new) - log P(old)`!
    """
    n = len(prompts)
    em_new = [0] * n
    em_old = [0] * n
    cand_hit = [0] * n
    lp_new_arr = np.zeros(n, dtype=np.float64)
    lp_old_arr = np.zeros(n, dtype=np.float64)
    gens = [""] * n

    subj_idxs = []
    seq_lens = []
    for p, sp in zip(prompts, char_spans):
        si, sl = locate_last_subject_token(tok, p, sp)
        subj_idxs.append(si)
        seq_lens.append(sl)

    handle = None
    ctx: Dict[str, Any] = {"active": False, "pos": None, "delta": None, "replace": None}
    if layer_idx is not None and (delta_mat is not None or replace_mat is not None):
        target_layer = get_decoder_layers(model)[layer_idx]

        def _patch_hook(_mod, _inp, out):
            if not ctx["active"]:
                return out
            is_tup = isinstance(out, tuple)
            h = out[0].clone() if is_tup else out.clone()
            if h.shape[1] <= 1:
                return out
            b_ar = torch.arange(h.shape[0], device=h.device)
            if ctx["replace"] is not None:
                h[b_ar, ctx["pos"], :] = ctx["replace"].to(device=h.device, dtype=h.dtype)
            elif ctx["delta"] is not None:
                h[b_ar, ctx["pos"], :] = h[b_ar, ctx["pos"], :] + ctx["delta"].to(device=h.device, dtype=h.dtype)
            return (h,) + out[1:] if is_tup else h

        handle = target_layer.register_forward_hook(_patch_hook)

    try:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            enc = tok(prompts[start:end], return_tensors="pt", padding=True, add_special_tokens=True)
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)
            t_pad = ids.shape[1]
            pos = [max(0, min(t_pad - 1, (t_pad - seq_lens[i]) + subj_idxs[i])) for i in range(start, end)]
            ctx["pos"] = torch.tensor(pos, dtype=torch.long, device=device)
            ctx["delta"] = (
                torch.from_numpy(delta_mat[start:end].astype(np.float32)).to(device)
                if delta_mat is not None else None
            )
            ctx["replace"] = (
                torch.from_numpy(replace_mat[start:end].astype(np.float32)).to(device)
                if replace_mat is not None else None
            )
            ctx["active"] = (handle is not None)
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=am, use_cache=True)
            ctx["active"] = False

            first_logits = out.logits[:, -1, :].float()
            first_lp = F.log_softmax(first_logits, dim=-1).cpu().numpy()
            gen_ids = _greedy_decode_from_prefill(
                model, out.logits[:, -1, :], out.past_key_values, am, max_new_tokens=max_new_tokens
            )
            for bi in range(end - start):
                idx = start + bi
                g_txt = tok.decode(gen_ids[bi].tolist(), skip_special_tokens=True)
                gens[idx] = g_txt
                if check_answer_match(g_txt, target_aliases_list[idx]):
                    em_new[idx] = 1
                if old_aliases_list is not None and check_answer_match(g_txt, old_aliases_list[idx]):
                    em_old[idx] = 1

                # Best first-token log-prob across target aliases (and old aliases)
                t_ids = [_first_tok_id(tok, a) for a in target_aliases_list[idx] if a]
                lp_new_arr[idx] = float(max(first_lp[bi, tid] for tid in t_ids)) if t_ids else -100.0
                if old_aliases_list is not None:
                    o_ids = [_first_tok_id(tok, a) for a in old_aliases_list[idx] if a]
                    lp_old_arr[idx] = float(max(first_lp[bi, tid] for tid in o_ids)) if o_ids else -100.0
                if candidate_pool_list is not None:
                    best_c = None
                    best_s = -1e9
                    for cand in candidate_pool_list[idx]:
                        cid = _first_tok_id(tok, cand)
                        sc = float(first_lp[bi, cid])
                        if sc > best_s:
                            best_s = sc
                            best_c = cand
                    if best_c is not None and any(
                        best_c.strip().lower() == a.strip().lower() for a in target_aliases_list[idx]
                    ):
                        cand_hit[idx] = 1
    finally:
        if handle is not None:
            handle.remove()

    return {
        "em_new": em_new,
        "em_old": em_old,
        "cand_hit": cand_hit,
        "lp_new": lp_new_arr.tolist(),
        "lp_old": lp_old_arr.tolist(),
        "gens": gens,
    }


# ======================================================================================
# 4. Leak-Free Multi-Layer Unlabeled Calibration Map Builder
# ======================================================================================

def build_leak_free_calibration_maps(
    model_A: Any,
    tok_A: Any,
    model_B: Any,
    tok_B: Any,
    device: torch.device,
    n_cal: int = 2500,
) -> Tuple[Dict[Tuple[int, int], dict], Dict[Tuple[int, int], dict], dict]:
    """
    Build leak-free full-rank Procrustes maps (residual stream) AND post-RMSNorm MLP-input
    sockets at ALL swept layer pairs [(11,6), (17,9), (20,11)] from Pilot 1's unlabeled
    entity-mention prompts, STRICTLY excluding every entity/country/capital/currency/language
    appearing in T0/T1/T2/T3!
    At (17, 9), also loads Pilot 1's saved 10k maps (reporting exact word overlap audit!).
    """
    forbidden_words = set()
    def _add(s: str):
        for w in re.findall(r"[a-z]+", s.lower()):
            if len(w) >= 3:
                forbidden_words.add(w)
    for c, info in COUNTRIES_INFO.items():
        _add(c)
        for a in info["aliases"]: _add(a)
        for k in ("capital", "currency", "language", "continent"):
            _add(info[k][0])
            for a in info[k][1]: _add(a)
    for lm in LANDMARKS_T0 + PEOPLE_FALLBACK_T0:
        _add(lm["name"])
    for ent, _, c in T3_ENTITIES:
        _add(ent)
        _add(c)

    # Load Pilot 1 unlabeled prompts (10,000 clean narrative entity sentences)
    pack = torch.load(PILOT1_UNLAB_PATH, map_location="cpu", weights_only=False)
    all_items = pack["train_items"]
    n_overlap_p1_10k = sum(1 for it in all_items if it["entity"].lower() in forbidden_words)
    clean_items = [it for it in all_items if it["entity"].lower() not in forbidden_words]
    cal_items = clean_items[:n_cal]
    cal_prompts = [it["prompt"] for it in cal_items]
    cal_spans = [tuple(it["char_span"]) for it in cal_items]

    layers_A = sorted({la for la, _ in SWEPT_LAYER_PAIRS})
    layers_B = sorted({lb for _, lb in SWEPT_LAYER_PAIRS})

    print(f"[cal] Extracting {len(cal_prompts)} leak-free unlabeled spans across A layers {layers_A} & B layers {layers_B}...")
    t0 = time.time()
    st_A = extract_multi_layer_states(model_A, tok_A, cal_prompts, cal_spans, layers_A, device, capture_mlp_in=True, batch_size=48)
    st_B = extract_multi_layer_states(model_B, tok_B, cal_prompts, cal_spans, layers_B, device, capture_mlp_in=True, batch_size=48)
    print(f"[cal] Extraction done in {time.time() - t0:.1f}s")

    resid_maps: Dict[Tuple[int, int], dict] = {}
    mlp_sockets: Dict[Tuple[int, int], dict] = {}
    for la, lb in SWEPT_LAYER_PAIRS:
        resid_maps[(la, lb)] = fit_procrustes_f64(st_A["resid"][la], st_B["resid"][lb])
        mlp_sockets[(la, lb)] = fit_platonic_socket_f64(st_A["mlp_in"][la], st_B["mlp_in"][lb])
        r_corr = mlp_sockets[(la, lb)]["r_corr"]
        print(
            f"  [cal ({la}->{lb})] resid_s={resid_maps[(la, lb)]['s']:.4f} | "
            f"mlp_in top-10 r_corr={float(np.mean(r_corr[:10])):.3f}, "
            f"top-64={float(np.mean(r_corr[:64])):.3f}, top-256={float(np.mean(r_corr[:256])):.3f}"
        )

    # Also attach Pilot 1's cached N=10,000 maps for (17, 9)
    p1_maps = torch.load(PILOT1_MAPS_PATH, map_location="cpu", weights_only=False)
    cal_meta = {
        "n_cal_leak_free": len(cal_prompts),
        "pilot1_10k_word_overlap_with_graft_pool": int(n_overlap_p1_10k),
        "swept_layer_pairs": SWEPT_LAYER_PAIRS,
        "pilot1_10k_available": True,
    }
    return resid_maps, mlp_sockets, {"cal_meta": cal_meta, "pilot1_10k": p1_maps}


# ======================================================================================
# 5. Sub-Tests T0, T1, T2 (+ Gate 3), T3
# ======================================================================================

def _landmark_prompts_and_targets(lm: dict, country_override: Optional[str] = None) -> Dict[str, dict]:
    """Exact shared-prefix prompts for all 4 geographic + 2 protected attributes."""
    c_key = country_override if country_override is not None else lm["country"]
    c_info = COUNTRIES_INFO[c_key]
    pref = lm["prefix"]
    cspan = (0, len(pref))
    return {
        "country": {"prompt": f"{pref} is located in the country of", "span": cspan, "aliases": c_info["aliases"]},
        "capital": {"prompt": f"{pref} is located in a country whose capital city is", "span": cspan, "aliases": c_info["capital"][1]},
        "currency": {"prompt": f"{pref} is located in a country whose official currency is the", "span": cspan, "aliases": c_info["currency"][1]},
        "language": {"prompt": f"{pref} is located in a country whose primary official language is", "span": cspan, "aliases": c_info["language"][1]},
        "prot_type": {"prompt": f"{pref} is a famous", "span": cspan, "aliases": lm["struct_aliases"]},
        "prot_fact": {"prompt": f"{pref}{lm['prot2_suffix']}", "span": cspan, "aliases": lm["prot2_aliases"]},
    }


def run_t0_competence_filter(
    model_A: Any, tok_A: Any, model_B: Any, tok_B: Any, device: torch.device
) -> Tuple[List[dict], dict]:
    """Screen landmarks (with people fallback if <12 pass) on unedited B and A."""
    print("\n=== T0: Unedited Knowledge & Competence Filter ===")
    attr_keys = ["country", "capital", "currency", "language", "prot_type", "prot_fact"]

    def _screen_pool(pool: List[dict], pool_name: str) -> Tuple[List[dict], dict]:
        per_item = []
        for mk, mdl, tk in (("B_smollm2", model_B, tok_B), ("A_qwen3b", model_A, tok_A)):
            prompts, spans, aliases = [], [], []
            for lm in pool:
                pm = _landmark_prompts_and_targets(lm)
                for ak in attr_keys:
                    prompts.append(pm[ak]["prompt"])
                    spans.append(pm[ak]["span"])
                    aliases.append(pm[ak]["aliases"])
            res = eval_prompts_patched(mdl, tk, prompts, spans, aliases, device=device, batch_size=32)
            for idx, lm in enumerate(pool):
                if mk == "B_smollm2":
                    per_item.append({"name": lm["name"], "country": lm["country"], "B": {}, "A": {}})
                tag = "B" if mk == "B_smollm2" else "A"
                for a_i, ak in enumerate(attr_keys):
                    k = idx * len(attr_keys) + a_i
                    per_item[idx][tag][ak] = {"ok": int(res["em_new"][k]), "gen": res["gens"][k]}

        passing = []
        for idx, rec in enumerate(per_item):
            b_geo = sum(rec["B"][k]["ok"] for k in ("country", "capital", "currency", "language"))
            b_prot = sum(rec["B"][k]["ok"] for k in ("prot_type", "prot_fact"))
            a_geo = sum(rec["A"][k]["ok"] for k in ("country", "capital", "currency", "language"))
            rec["B_geo_score"] = b_geo
            rec["B_prot_score"] = b_prot
            rec["A_geo_score"] = a_geo
            # Pass if B gets >=3/4 geo AND >=1/2 protected (AND B knows country!)
            rec["passed"] = bool(rec["B"]["country"]["ok"] == 1 and b_geo >= 3 and b_prot >= 1)
            if rec["passed"]:
                passing.append(pool[idx])

        summary = {
            "pool_name": pool_name,
            "n_total": len(pool),
            "n_passed": len(passing),
            "B_mean_by_attr": {ak: round(float(np.mean([r["B"][ak]["ok"] for r in per_item])), 4) for ak in attr_keys},
            "A_mean_by_attr": {ak: round(float(np.mean([r["A"][ak]["ok"] for r in per_item])), 4) for ak in attr_keys},
            "items": per_item,
        }
        print(
            f"  [T0 {pool_name}] passed={len(passing)}/{len(pool)} | "
            f"B_acc={summary['B_mean_by_attr']} | A_acc={summary['A_mean_by_attr']}"
        )
        return passing, summary

    passing_lm, sum_lm = _screen_pool(LANDMARKS_T0, "landmarks_32")
    t0_res = {"landmarks": sum_lm, "used_fallback_people": False}
    if len(passing_lm) >= 10:
        return passing_lm, t0_res
    # If SmolLM2 passes <10 landmarks, also screen people fallback and merge / switch!
    passing_ppl, sum_ppl = _screen_pool(PEOPLE_FALLBACK_T0, "people_fallback_20")
    t0_res["people_fallback"] = sum_ppl
    if len(passing_ppl) > len(passing_lm):
        t0_res["used_fallback_people"] = True
        return passing_ppl, t0_res
    return passing_lm, t0_res


def run_t1_forced_gate_audit(
    passing_items: List[dict],
    model_A: Any,
    tok_A: Any,
    model_B: Any,
    tok_B: Any,
    resid_maps: Dict[Tuple[int, int], dict],
    device: torch.device,
) -> dict:
    """
    T1: Forced-Gate Value, Multi-Hop Ripple & Orthogonal Retention Audit across
    3 depths {(11,6), (17,9), (20,11)} x scales {0.5, 1.0, 2.0, 3.0} x 7 vector families!
    """
    print(f"\n=== T1: Forced-Gate Value, Multi-Hop Ripple & Retention Audit (n={len(passing_items)}) ===")
    derangement = assign_deranged_targets(passing_items, seed=SEED)
    used_countries = sorted({it["country"] for it in passing_items} | set(derangement.values()))

    # Candidate pools for candidate-choice accuracy
    cand_pools = {
        "country": [c for c in used_countries],
        "capital": [COUNTRIES_INFO[c]["capital"][0] for c in used_countries],
        "currency": sorted({COUNTRIES_INFO[c]["currency"][0] for c in used_countries}),
        "language": sorted({COUNTRIES_INFO[c]["language"][0] for c in used_countries}),
    }

    # 1. Extract neutral country & capital noun vectors u(c) and u(cap) on BOTH A and B across all 3 layers!
    noun_templates = [
        ("The country of {x} is known", "country"),
        ("Geographically, {x} is situated", "country"),
        ("In international affairs, {x} plays", "country"),
        ("The nation of {x} has", "country"),
    ]
    cap_templates = [
        ("The capital city of {x} is known", "capital"),
        ("Travelers visiting {x} often see", "capital"),
        ("Historically, {x} has been", "capital"),
    ]
    layers_A = sorted({la for la, _ in SWEPT_LAYER_PAIRS})
    layers_B = sorted({lb for _, lb in SWEPT_LAYER_PAIRS})

    c_prompts, c_spans, c_keys = [], [], []
    for c in used_countries:
        for tmpl, _ in noun_templates:
            p = tmpl.format(x=c)
            i0 = p.index(c)
            c_prompts.append(p)
            c_spans.append((i0, i0 + len(c)))
            c_keys.append(("country", c))
        cap = COUNTRIES_INFO[c]["capital"][0]
        for tmpl, _ in cap_templates:
            p = tmpl.format(x=cap)
            i0 = p.index(cap)
            c_prompts.append(p)
            c_spans.append((i0, i0 + len(cap)))
            c_keys.append(("capital", c))

    st_noun_A = extract_multi_layer_states(model_A, tok_A, c_prompts, c_spans, layers_A, device)["resid"]
    st_noun_B = extract_multi_layer_states(model_B, tok_B, c_prompts, c_spans, layers_B, device)["resid"]

    # Average templates -> u_A[la][kind][c], u_B[lb][kind][c]
    def _avg_by_key(st_dict: Dict[int, np.ndarray]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        out: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
        for l, mat in st_dict.items():
            buckets: Dict[Tuple[str, str], List[np.ndarray]] = defaultdict(list)
            for idx, key in enumerate(c_keys):
                buckets[key].append(mat[idx].astype(np.float64))
            out[l] = {"country": {}, "capital": {}}
            for (kind, c), vecs in buckets.items():
                out[l][kind][c] = np.mean(vecs, axis=0)
        return out

    u_A = _avg_by_key(st_noun_A)
    u_B = _avg_by_key(st_noun_B)

    # 2. Extract landmark states across ALL 32 landmarks (for leave-target-out class-mean diffs!)
    lm_pool_all = LANDMARKS_T0 if passing_items[0] in LANDMARKS_T0 else PEOPLE_FALLBACK_T0
    lm_prompts = [f"{lm['prefix']} is located in" for lm in lm_pool_all]
    lm_spans = [(0, len(lm["prefix"])) for lm in lm_pool_all]
    st_lm_A = extract_multi_layer_states(model_A, tok_A, lm_prompts, lm_spans, layers_A, device)["resid"]
    st_lm_B = extract_multi_layer_states(model_B, tok_B, lm_prompts, lm_spans, layers_B, device)["resid"]
    lm_name_to_idx = {lm["name"]: i for i, lm in enumerate(lm_pool_all)}

    def _loo_class_mean(st_mat: np.ndarray, country: str, exclude_name: str) -> np.ndarray:
        idxs = [i for i, lm in enumerate(lm_pool_all) if lm["country"] == country and lm["name"] != exclude_name]
        if not idxs:
            idxs = [i for i, lm in enumerate(lm_pool_all) if lm["country"] == country]
        return np.mean(st_mat[idxs].astype(np.float64), axis=0)

    # 3. Build evaluation prompt table for the passing landmarks (6 prompts per landmark: 4 geo + 2 prot)
    attr_keys = ["country", "capital", "currency", "language", "prot_type", "prot_fact"]
    eval_prompts, eval_spans, eval_new_al, eval_old_al, eval_cands, eval_meta = [], [], [], [], [], []
    for lm_i, lm in enumerate(passing_items):
        c_old = lm["country"]
        c_new = derangement[lm["name"]]
        pm_old = _landmark_prompts_and_targets(lm, country_override=c_old)
        pm_new = _landmark_prompts_and_targets(lm, country_override=c_new)
        for ak in attr_keys:
            eval_prompts.append(pm_old[ak]["prompt"])
            eval_spans.append(pm_old[ak]["span"])
            eval_new_al.append(pm_new[ak]["aliases"])
            eval_old_al.append(pm_old[ak]["aliases"])
            eval_cands.append(cand_pools[ak] if ak in cand_pools else [])
            eval_meta.append({"lm_idx": lm_i, "name": lm["name"], "attr": ak, "c_old": c_old, "c_new": c_new})

    # Unpatched baseline on B (gives baseline log-odds & baseline protected correctness!)
    base_res = eval_prompts_patched(
        model_B, tok_B, eval_prompts, eval_spans, eval_new_al,
        old_aliases_list=eval_old_al, candidate_pool_list=[c if c else None for c in eval_cands] if False else None,
        device=device, batch_size=32,
    )
    # Re-score candidate choice on geo prompts cleanly
    base_lp_new = np.array(base_res["lp_new"], dtype=np.float64)
    base_lp_old = np.array(base_res["lp_old"], dtype=np.float64)
    base_prot_ok = [
        (i, base_res["em_old"][i]) for i, m in enumerate(eval_meta) if m["attr"] in ("prot_type", "prot_fact")
    ]
    base_prot_correct_idxs = [i for i, ok in base_prot_ok if ok == 1]

    # Deterministic random orthogonal matrix for shuffled alignment control & random directions
    rng = np.random.default_rng(SEED + 777)
    R_rand_ctrl = _polar_f64(rng.standard_normal((2048, 2048)))

    scales = [0.5, 1.0, 2.0, 3.0]
    t1_results: Dict[str, Any] = {
        "n_landmarks": len(passing_items),
        "derangement": derangement,
        "n_baseline_correct_protected_prompts": len(base_prot_correct_idxs),
        "by_layer_pair": {},
    }

    for la, lb in SWEPT_LAYER_PAIRS:
        mp = resid_maps[(la, lb)]
        R, s_map = mp["R"], mp["s"]
        layer_tag = f"lA{la}_to_lB{lb}"
        t1_results["by_layer_pair"][layer_tag] = {}

        # Build the 7 base unit-scale delta matrices [n_landmarks, D] for this layer pair
        vec_families: Dict[str, np.ndarray] = {}
        d_nat_noun = np.zeros((len(passing_items), 2048), dtype=np.float64)
        d_soc_noun = np.zeros((len(passing_items), 2048), dtype=np.float64)
        d_nat_cls = np.zeros((len(passing_items), 2048), dtype=np.float64)
        d_soc_cls = np.zeros((len(passing_items), 2048), dtype=np.float64)
        d_sib_nat = np.zeros((len(passing_items), 2048), dtype=np.float64)
        d_sib_soc = np.zeros((len(passing_items), 2048), dtype=np.float64)
        d_rand = np.zeros((len(passing_items), 2048), dtype=np.float64)
        d_wrong_c = np.zeros((len(passing_items), 2048), dtype=np.float64)
        d_shuf_al = np.zeros((len(passing_items), 2048), dtype=np.float64)

        norm_ratios = []
        for i, lm in enumerate(passing_items):
            c_old, c_new = lm["country"], derangement[lm["name"]]
            dn_b = u_B[lb]["country"][c_new] - u_B[lb]["country"][c_old]
            dn_a = u_A[la]["country"][c_new] - u_A[la]["country"][c_old]
            dn_a_rot = s_map * (dn_a @ R.T)
            n_b = np.linalg.norm(dn_b)
            n_ar = np.linalg.norm(dn_a_rot)
            norm_ratios.append(float(n_ar / max(n_b, 1e-12)))

            d_nat_noun[i] = dn_b
            # Residual-norm matched socket noun diff (exact fair scale comparison vs Native-B!)
            d_soc_noun[i] = dn_a_rot * (n_b / max(n_ar, 1e-12))

            dc_b = _loo_class_mean(st_lm_B[lb], c_new, lm["name"]) - _loo_class_mean(st_lm_B[lb], c_old, lm["name"])
            dc_a = _loo_class_mean(st_lm_A[la], c_new, lm["name"]) - _loo_class_mean(st_lm_A[la], c_old, lm["name"])
            dc_a_rot = s_map * (dc_a @ R.T)
            d_nat_cls[i] = dc_b
            d_soc_cls[i] = dc_a_rot * (np.linalg.norm(dc_b) / max(np.linalg.norm(dc_a_rot), 1e-12))

            cap_b = u_B[lb]["capital"][c_new] - u_B[lb]["capital"][c_old]
            cap_a = u_A[la]["capital"][c_new] - u_A[la]["capital"][c_old]
            cap_a_rot = s_map * (cap_a @ R.T)
            d_sib_nat[i] = dn_b + cap_b
            d_sib_soc[i] = (dn_a_rot + cap_a_rot) * (np.linalg.norm(d_sib_nat[i]) / max(np.linalg.norm(dn_a_rot + cap_a_rot), 1e-12))

            rv = rng.standard_normal(2048)
            d_rand[i] = rv * (n_b / max(np.linalg.norm(rv), 1e-12))
            wrong_cands = [c for c in used_countries if c != c_old and c != c_new]
            c_wrong = wrong_cands[i % len(wrong_cands)]
            dw_b = u_B[lb]["country"][c_wrong] - u_B[lb]["country"][c_old]
            d_wrong_c[i] = dw_b * (n_b / max(np.linalg.norm(dw_b), 1e-12))
            sh_v = s_map * (dn_a @ R_rand_ctrl.T)
            d_shuf_al[i] = sh_v * (n_b / max(np.linalg.norm(sh_v), 1e-12))

        vec_families = {
            "native_B_noun_diff": d_nat_noun,
            "socket_noun_diff": d_soc_noun,
            "native_B_class_mean_diff": d_nat_cls,
            "socket_class_mean_diff": d_soc_cls,
            "city_country_sibling_native_B": d_sib_nat,
            "city_country_sibling_socket": d_sib_soc,
            "wrong_country_control": d_wrong_c,
            "norm_matched_random_dir": d_rand,
            "shuffled_alignment_control": d_shuf_al,
        }
        t1_results["by_layer_pair"][layer_tag]["mean_raw_socket_to_native_norm_ratio"] = round(float(np.mean(norm_ratios)), 4)

        for vname, V_lm in vec_families.items():
            # Expand [n_lm, D] -> [n_lm * 6, D] matching eval_prompts
            V_exp = np.repeat(V_lm, len(attr_keys), axis=0)
            v_scales = [1.0, 2.0] if vname in ("wrong_country_control", "norm_matched_random_dir", "shuffled_alignment_control") else scales
            t1_results["by_layer_pair"][layer_tag][vname] = {}

            for sc in v_scales:
                ev = eval_prompts_patched(
                    model_B, tok_B, eval_prompts, eval_spans, eval_new_al,
                    old_aliases_list=eval_old_al,
                    candidate_pool_list=[c if c else ["dummy"] for c in eval_cands],
                    layer_idx=lb, delta_mat=(sc * V_exp), device=device, batch_size=32,
                )
                lp_new = np.array(ev["lp_new"], dtype=np.float64)
                lp_old = np.array(ev["lp_old"], dtype=np.float64)

                by_attr = {}
                for ak in ("country", "capital", "currency", "language"):
                    idxs = [i for i, m in enumerate(eval_meta) if m["attr"] == ak]
                    by_attr[ak] = {
                        "em_new": round(float(np.mean([ev["em_new"][i] for i in idxs])), 4),
                        "em_old": round(float(np.mean([ev["em_old"][i] for i in idxs])), 4),
                        "cand_choice": round(float(np.mean([ev["cand_hit"][i] for i in idxs])), 4),
                        "log_odds_shift": round(
                            float(np.mean((lp_new[idxs] - lp_old[idxs]) - (base_lp_new[idxs] - base_lp_old[idxs]))), 4
                        ),
                    }
                ripple_em = float(np.mean([by_attr[ak]["em_new"] for ak in ("capital", "currency", "language")]))
                ripple_cand = float(np.mean([by_attr[ak]["cand_choice"] for ak in ("capital", "currency", "language")]))
                # Joint 4/4 flip rate per landmark
                joint4 = []
                for lm_i in range(len(passing_items)):
                    ok4 = all(ev["em_new"][lm_i * 6 + a_i] == 1 for a_i in range(4))
                    joint4.append(int(ok4))
                # Protected retention on baseline-correct protected prompts
                prot_ret = (
                    float(np.mean([ev["em_old"][i] for i in base_prot_correct_idxs]))
                    if base_prot_correct_idxs else None
                )
                prot_lp_delta = (
                    float(np.mean(lp_old[base_prot_correct_idxs] - base_lp_old[base_prot_correct_idxs]))
                    if base_prot_correct_idxs else None
                )

                rec = {
                    "scale": sc,
                    "by_attr": by_attr,
                    "unseen_ripple_em_3attr": round(ripple_em, 4),
                    "unseen_ripple_cand_3attr": round(ripple_cand, 4),
                    "joint_4of4_flip_rate": round(float(np.mean(joint4)), 4),
                    "protected_em_retention": round(prot_ret, 4) if prot_ret is not None else None,
                    "protected_logp_delta": round(prot_lp_delta, 4) if prot_lp_delta is not None else None,
                }
                t1_results["by_layer_pair"][layer_tag][vname][str(sc)] = rec
                print(
                    f"  [T1 {layer_tag} {vname:30s} s={sc:.1f}] "
                    f"ctry={by_attr['country']['em_new']:.2f} cap={by_attr['capital']['em_new']:.2f} "
                    f"cur={by_attr['currency']['em_new']:.2f} lng={by_attr['language']['em_new']:.2f} | "
                    f"ripple3={ripple_em:.3f} (cand={ripple_cand:.3f}) joint4={rec['joint_4of4_flip_rate']:.3f} | "
                    f"prot_ret={rec['protected_em_retention']} (dlp={rec['protected_logp_delta']})"
                )

    return t1_results


def run_t2_key_selectivity_and_gate3(
    passing_items: List[dict],
    dataset: dict,
    model_A: Any,
    tok_A: Any,
    model_B: Any,
    tok_B: Any,
    mlp_sockets: Dict[Tuple[int, int], dict],
    device: torch.device,
) -> dict:
    """
    T2: Key Selectivity & Hard-Neighbor Margin across k in {64, 128, 256, 512, 2048}
        at (17 -> 9) [plus (20 -> 11) check] on post-RMSNorm MLP-input states,
        PLUS Gate 3 implicit-constant sigma_c stability audit!
    """
    print(f"\n=== T2: Key Selectivity, Hard-Neighbor Margin & Gate 3 sigma_c (n={len(passing_items)}) ===")

    fit_templates = [
        "{P} is located in a well-known region.",
        "Historically, {P} became a major landmark.",
        "Visitors often travel to see {P} each year.",
        "Which country contains {P} today?",
        "In architecture books, {P} is frequently cited.",
        "Many photographs feature {P} at sunset.",
    ]
    unseen_paraphrase_templates = [
        "Built long ago, {P} attracts millions of tourists.",
        "What currency is used where {P} stands?",
        "Although many people have heard of {P}, few know its full story.",
        "Among world heritage sites, {P} remains unforgettable.",
        "Tour guides routinely highlight {P} during walking tours.",
        "A documentary about {P} aired last night.",
    ]
    # Hard negatives: ALL landmarks in LANDMARKS_T0 + extra shared-head-noun traps + sibling entities
    extra_traps = [
        "CN Tower", "Space Needle", "Willis Tower", "Burj Khalifa", "Sydney Opera House",
        "Gustave Eiffel", "Paris", "France", "Tokyo", "Japan", "London", "Rome",
        "Berlin", "New Delhi", "Beijing", "Moscow", "Washington", "Brasilia",
    ]
    all_lm_names = [lm["name"] for lm in LANDMARKS_T0]
    neg_entities = sorted(set(all_lm_names + extra_traps))
    neg_templates = [
        "Visitors often travel to see {E} each year.",
        "In travel guides, {E} is frequently mentioned.",
    ]

    # Background tokens from 200 dataset.json train prompts (ALL tokens!)
    bg_prompts = [f["direct_prompt"] for f in dataset["facts"] if f["split"] == "train"][:200]

    # Build fit, unseen-pos, and hard-neg prompt lists
    fit_p, fit_sp, fit_lm = [], [], []
    for lm in passing_items:
        for tmpl in fit_templates:
            p = tmpl.format(P=lm["prefix"])
            i0 = p.index(lm["prefix"])
            fit_p.append(p)
            fit_sp.append((i0, i0 + len(lm["prefix"])))
            fit_lm.append(lm["name"])

    pos_p, pos_sp, pos_lm = [], [], []
    for lm in passing_items:
        for tmpl in unseen_paraphrase_templates:
            p = tmpl.format(P=lm["prefix"])
            i0 = p.index(lm["prefix"])
            pos_p.append(p)
            pos_sp.append((i0, i0 + len(lm["prefix"])))
            pos_lm.append(lm["name"])
        # Also include lowercase variation as an unseen stress paraphrase!
        p_low = f"During our trip we visited {lm['prefix'].lower()} in the afternoon."
        i0 = p_low.index(lm["prefix"].lower())
        pos_p.append(p_low)
        pos_sp.append((i0, i0 + len(lm["prefix"])))
        pos_lm.append(lm["name"])

    neg_p, neg_sp, neg_ent = [], [], []
    for e in neg_entities:
        for tmpl in neg_templates:
            p = tmpl.format(E=e)
            i0 = p.index(e)
            neg_p.append(p)
            neg_sp.append((i0, i0 + len(e)))
            neg_ent.append(e)

    # Extract post-RMSNorm MLP-input states at (17, 9) and (20, 11)
    t2_pairs = [(17, 9), (20, 11)]
    la_list = [p[0] for p in t2_pairs]
    lb_list = [p[1] for p in t2_pairs]

    H_fit_A = extract_multi_layer_states(model_A, tok_A, fit_p, fit_sp, la_list, device, capture_mlp_in=True)["mlp_in"]
    H_fit_B = extract_multi_layer_states(model_B, tok_B, fit_p, fit_sp, lb_list, device, capture_mlp_in=True)["mlp_in"]
    H_pos_B = extract_multi_layer_states(model_B, tok_B, pos_p, pos_sp, lb_list, device, capture_mlp_in=True)["mlp_in"]
    H_neg_B = extract_multi_layer_states(model_B, tok_B, neg_p, neg_sp, lb_list, device, capture_mlp_in=True)["mlp_in"]

    t2_out: Dict[str, Any] = {"by_layer_pair": {}}

    for la, lb in t2_pairs:
        tag = f"lA{la}_to_lB{lb}"
        sock = mlp_sockets[(la, lb)]
        U_A, U_B = sock["U_A"], sock["U_B"]
        mu_A, mu_B = sock["mu_A"], sock["mu_B"]
        r_corr = sock["r_corr"]

        # Extract ALL background tokens on B at layer lb
        bg_nonbos, bg_bos = extract_all_tokens_mlp_in(model_B, tok_B, bg_prompts, lb, device, batch_size=32)
        n_bg = len(bg_nonbos)
        mid_bg = n_bg // 2
        bg_tr, bg_ev = bg_nonbos[:mid_bg], bg_nonbos[mid_bg:]

        # Gate 3: Implicit-constant sigma_c audit (separately non-BOS vs with-BOS!)
        gate3_nonbos = fit_implicit_constant_gate3(bg_tr, np.vstack([bg_ev, H_pos_B[lb], H_neg_B[lb]]))
        gate3_with_bos = fit_implicit_constant_gate3(
            np.vstack([bg_tr, bg_bos[: len(bg_bos) // 2]]),
            np.vstack([bg_ev, bg_bos[len(bg_bos) // 2 :], H_pos_B[lb]]),
        )
        print(
            f"  [Gate 3 {tag}] non-BOS: sigma_c={gate3_nonbos['eval_std_sigma_c']:.4f}, "
            f"delta99={gate3_nonbos['eval_p99_abs_err_delta99']:.4f} | "
            f"with-BOS: sigma_c={gate3_with_bos['eval_std_sigma_c']:.4f}, "
            f"delta99={gate3_with_bos['eval_p99_abs_err_delta99']:.4f}"
        )

        # Native-B background covariance C_bg (centered on bg_tr)
        bg_tr_c = bg_tr.astype(np.float64) - mu_B
        C_bg_full = (bg_tr_c.T @ bg_tr_c) / max(len(bg_tr_c), 1)

        # Group fit/pos/neg indices by entity name
        fit_by_lm = defaultdict(list)
        for i, nm in enumerate(fit_lm): fit_by_lm[nm].append(i)
        pos_by_lm = defaultdict(list)
        for i, nm in enumerate(pos_lm): pos_by_lm[nm].append(i)
        neg_by_ent = defaultdict(list)
        for i, e in enumerate(neg_ent): neg_by_ent[e].append(i)

        # Precompute centered matrices on B & A
        fit_A_c = H_fit_A[la].astype(np.float64) - mu_A
        fit_B_c = H_fit_B[lb].astype(np.float64) - mu_B
        pos_B_c = H_pos_B[lb].astype(np.float64) - mu_B
        neg_B_c = H_neg_B[lb].astype(np.float64) - mu_B
        bg_ev_c = bg_ev.astype(np.float64) - mu_B

        # Precompute Native-B full-space MVDR d'^2 as denominator for d'^2(P^k)/d'^2(native)
        k_sweep = [64, 128, 256, 512, 2048]
        layer_res: Dict[str, Any] = {
            "n_bg_tokens_train": int(len(bg_tr)),
            "n_bg_tokens_eval": int(len(bg_ev)),
            "gate3_sigma_c_nonbos": gate3_nonbos,
            "gate3_sigma_c_with_bos": gate3_with_bos,
            "by_k": {},
        }

        native_full_dprime_sq = None

        # Evaluate in descending k order so k=2048 native reference is computed first!
        for k in reversed(k_sweep):
            UA_k = U_A[:, :k]
            UB_k = U_B[:, :k]
            rk = r_corr[:k]
            # Project into P^k
            Z_fit_A = fit_A_c @ UA_k
            Z_fit_B = fit_B_c @ UB_k
            Z_pos_B = pos_B_c @ UB_k
            Z_neg_B = neg_B_c @ UB_k
            Z_bg_tr = bg_tr_c @ UB_k
            Z_bg_ev = bg_ev_c @ UB_k

            # Background covariance in P^k
            C_pk = (Z_bg_tr.T @ Z_bg_tr) / max(len(Z_bg_tr), 1)
            tr_pk = max(float(np.trace(C_pk)) / k, 1e-12)
            C_pk_reg = 0.5 * (C_pk + C_pk.T) + (0.05 * tr_pk) * np.eye(k, dtype=np.float64)
            c_pk_fac = sla.cho_factor(C_pk_reg, check_finite=False)

            # Evaluate 4 key recipes:
            #   1. donor_Pk_raw_cos
            #   2. donor_Pk_whitened (C_pk^{-1} z_A)
            #   3. donor_Pk_whitened_r2 (r_i^2 Wiener-weighted + whitened!)
            #   4. native_B_mvdr (Fable's C^{-1}K(K^T C^{-1}K)^{-1}e_1 on B inside P^k!)
            recipes = ["donor_Pk_raw_cos", "donor_Pk_whitened", "donor_Pk_whitened_r2", "native_B_mvdr"]
            rec_stats: Dict[str, Any] = {}

            for rname in recipes:
                entity_margins = []
                all_pos_s = []
                all_hneg_s = []
                all_bg_s = []
                dprimes = []

                for lm in passing_items:
                    nm = lm["name"]
                    p_idxs = pos_by_lm[nm]
                    # Hard negatives = all neg_entities EXCEPT nm itself!
                    hn_idxs = [i for e, idxs in neg_by_ent.items() if e != nm for i in idxs]
                    # Mean hard-negative prototypes for MVDR (same-country landmarks + top traps, excluding nm)
                    hn_proto_list = [
                        np.mean(Z_neg_B[idxs], axis=0)
                        for e, idxs in neg_by_ent.items()
                        if e != nm and (e in extra_traps or any(x["name"] == e and x["country"] == lm["country"] for x in LANDMARKS_T0))
                    ]
                    hn_protos = np.vstack(hn_proto_list) if hn_proto_list else np.zeros((0, k))

                    if rname == "donor_Pk_raw_cos":
                        g = np.mean(Z_fit_A[fit_by_lm[nm]], axis=0)
                        g = g / max(np.linalg.norm(g), 1e-12)
                        def _score(M):
                            return (M @ g) / np.maximum(np.linalg.norm(M, axis=1), 1e-12)
                    elif rname == "donor_Pk_whitened":
                        z_m = np.mean(Z_fit_A[fit_by_lm[nm]], axis=0)
                        g = sla.cho_solve(c_pk_fac, z_m, check_finite=False)
                        # Scale so g^T z_m = 1 (exact linear pre-activation matched to SwiGLU W_gate h_tilde!)
                        g = g / max(float(g @ z_m), 1e-12)
                        def _score(M):
                            return M @ g
                    elif rname == "donor_Pk_whitened_r2":
                        z_m = np.mean(Z_fit_A[fit_by_lm[nm]], axis=0) * (rk ** 2)
                        g = sla.cho_solve(c_pk_fac, z_m, check_finite=False)
                        g = g / max(float(g @ z_m), 1e-12)
                        def _score(M):
                            return M @ g
                    else:  # native_B_mvdr (Fable's exact C^{-1}K(K^T C^{-1}K)^{-1}e_1 linear filter!)
                        z_m = np.mean(Z_fit_B[fit_by_lm[nm]], axis=0)
                        g_unit = compute_mvdr_filter(z_m, hn_protos, C_pk, shrink_rel=0.05)
                        g = g_unit / max(float(g_unit @ z_m), 1e-12)
                        def _score(M):
                            return M @ g

                    sp = _score(Z_pos_B[p_idxs])
                    sh = _score(Z_neg_B[hn_idxs])
                    sb = _score(Z_bg_ev)
                    entity_margins.append(float(np.min(sp) - np.max(sh)))
                    all_pos_s.append(sp)
                    all_hneg_s.append(sh)
                    all_bg_s.append(sb)
                    dprimes.append(compute_dprime_sq(sp, np.concatenate([sh, sb])))

                pos_all = np.concatenate(all_pos_s)
                hneg_all = np.concatenate(all_hneg_s)
                bg_all = np.concatenate(all_bg_s)
                neg_all = np.concatenate([hneg_all, bg_all])
                thr_01fpr = float(np.percentile(neg_all, 99.9))
                rec_at_01fpr = float(np.mean(pos_all > thr_01fpr))
                neon_margin = float(np.percentile(pos_all, 10) - np.percentile(hneg_all, 99))
                mean_dp2 = float(np.mean(dprimes))
                if k == 2048 and rname == "native_B_mvdr":
                    native_full_dprime_sq = mean_dp2

                # Gate 3 ratio: theta * delta99 / neon_margin
                g3_ratio = (
                    round(float(abs(thr_01fpr) * gate3_nonbos["eval_p99_abs_err_delta99"] / max(neon_margin, 1e-6)), 3)
                    if neon_margin > 0 else None
                )
                rec_stats[rname] = {
                    "mean_entity_margin_minpos_maxhneg": round(float(np.mean(entity_margins)), 4),
                    "frac_entities_positive_margin": round(float(np.mean([m > 0 for m in entity_margins])), 4),
                    "neon_margin_p10pos_minus_p99hneg": round(neon_margin, 4),
                    "recall_at_0_1pct_fpr": round(rec_at_01fpr, 4),
                    "thr_at_0_1pct_fpr": round(thr_01fpr, 4),
                    "mean_dprime_sq": round(mean_dp2, 3),
                    "dprime_sq_ratio_vs_native_full": round(mean_dp2 / max(native_full_dprime_sq or 1.0, 1e-12), 4),
                    "k_over_d": round(k / 2048.0, 4),
                    "gate3_theta_delta99_over_margin": g3_ratio,
                }
            layer_res["by_k"][str(k)] = rec_stats
            print(
                f"  [T2 {tag} k={k:4d}] "
                f"raw: m={rec_stats['donor_Pk_raw_cos']['neon_margin_p10pos_minus_p99hneg']:+.3f} R@0.1%={rec_stats['donor_Pk_raw_cos']['recall_at_0_1pct_fpr']:.2f} | "
                f"wh: m={rec_stats['donor_Pk_whitened']['neon_margin_p10pos_minus_p99hneg']:+.3f} R@0.1%={rec_stats['donor_Pk_whitened']['recall_at_0_1pct_fpr']:.2f} | "
                f"wh_r2: m={rec_stats['donor_Pk_whitened_r2']['neon_margin_p10pos_minus_p99hneg']:+.3f} R@0.1%={rec_stats['donor_Pk_whitened_r2']['recall_at_0_1pct_fpr']:.2f} dp2/nat={rec_stats['donor_Pk_whitened_r2']['dprime_sq_ratio_vs_native_full']:.3f} | "
                f"nat_MVDR: m={rec_stats['native_B_mvdr']['neon_margin_p10pos_minus_p99hneg']:+.3f} R@0.1%={rec_stats['native_B_mvdr']['recall_at_0_1pct_fpr']:.2f}"
            )
        t2_out["by_layer_pair"][tag] = layer_res

    return t2_out


def _t3_prompts_for_entity(ent: str, kind: str, country: str) -> Dict[str, dict]:
    c_info = COUNTRIES_INFO[country]
    if kind == "city":
        pref = f"The city of {ent}"
        c_prompt = f"{pref} is located in the country of"
    elif kind == "university":
        pref = f"The university {ent}" if not ent.startswith("University") else f"The {ent}"
        c_prompt = f"{pref} is located in the country of"
    elif kind == "river":
        pref = f"The {ent}"
        c_prompt = f"{pref} flows primarily through the country of"
    else:  # landmark
        pref = f"The landmark {ent}"
        c_prompt = f"{pref} is located in the country of"
    cspan = (0, len(pref))
    return {
        "country": {"prompt": c_prompt, "span": cspan, "aliases": c_info["aliases"]},
        "capital": {"prompt": f"{pref} is located in a country whose capital city is", "span": cspan, "aliases": c_info["capital"][1]},
        "currency": {"prompt": f"{pref} is located in a country whose official currency is the", "span": cspan, "aliases": c_info["currency"][1]},
        "language": {"prompt": f"{pref} is located in a country whose primary official language is", "span": cspan, "aliases": c_info["language"][1]},
        "continent": {"prompt": f"{pref} is located on the continent of", "span": cspan, "aliases": c_info["continent"][1]},
    }


def run_t3_knowledge_gap_transplant(
    model_A: Any,
    tok_A: Any,
    model_B: Any,
    tok_B: Any,
    resid_maps: Dict[Tuple[int, int], dict],
    pilot1_10k_maps: dict,
    device: torch.device,
) -> dict:
    """
    T3: Fable's High-Upside Knowledge-Gap Noun Transplant!
    Screen ~145 medium/long-tail entities for A-right (qwen2.5-3b), B-wrong (smollm2-1.7b)
    on `country`. Patch B's `last_subj` with A's rotated entity vector via Pilot 1's full-rank
    10k maps at (17 -> 9) and 2.5k map at (20 -> 11), testing `country` + 4 UNSEEN dependent
    attributes (`capital`, `currency`, `language`, `continent`), plus all 3 mandatory controls!
    """
    print(f"\n=== T3: Knowledge-Gap Noun Transplant (screening {len(T3_ENTITIES)} long-tail entities) ===")
    scr_prompts, scr_spans, scr_al = [], [], []
    for ent, kind, c in T3_ENTITIES:
        pm = _t3_prompts_for_entity(ent, kind, c)["country"]
        scr_prompts.append(pm["prompt"])
        scr_spans.append(pm["span"])
        scr_al.append(pm["aliases"])

    res_A = eval_prompts_patched(model_A, tok_A, scr_prompts, scr_spans, scr_al, device=device, batch_size=32)
    res_B = eval_prompts_patched(model_B, tok_B, scr_prompts, scr_spans, scr_al, device=device, batch_size=32)

    gap_items = []
    both_right_items = []
    for i, (ent, kind, c) in enumerate(T3_ENTITIES):
        a_ok = int(res_A["em_new"][i])
        b_ok = int(res_B["em_new"][i])
        if a_ok == 1 and b_ok == 0:
            gap_items.append({"idx": i, "entity": ent, "kind": kind, "country": c, "A_gen": res_A["gens"][i], "B_gen": res_B["gens"][i]})
        elif a_ok == 1 and b_ok == 1:
            both_right_items.append({"idx": i, "entity": ent, "kind": kind, "country": c})

    print(
        f"  [T3 screen] A_right={sum(res_A['em_new'])}/{len(T3_ENTITIES)}, "
        f"B_right={sum(res_B['em_new'])}/{len(T3_ENTITIES)}, "
        f"GAP (A-right, B-wrong)={len(gap_items)}, both_right={len(both_right_items)}"
    )
    if len(gap_items) < 3:
        return {"n_screened": len(T3_ENTITIES), "n_gap": len(gap_items), "error": "Too few gap entities"}

    # Assign a DIFFERENT gap entity e' (different country/capital/currency/language/continent where possible!) for Control 1
    rng = np.random.default_rng(SEED + 999)
    alt_idx_map = []
    for i, gi in enumerate(gap_items):
        ci = COUNTRIES_INFO[gi["country"]]
        cands = [
            j for j, gj in enumerate(gap_items)
            if gj["country"] != gi["country"]
            and COUNTRIES_INFO[gj["country"]]["capital"][0] != ci["capital"][0]
            and COUNTRIES_INFO[gj["country"]]["currency"][0] != ci["currency"][0]
        ]
        if not cands:
            cands = [j for j, gj in enumerate(gap_items) if gj["country"] != gi["country"]]
        alt_idx_map.append(int(cands[int(rng.integers(0, len(cands)))]) if cands else (i + 1) % len(gap_items))

    # Build 5-attribute prompts for all gap items
    t3_attrs = ["country", "capital", "currency", "language", "continent"]
    g_prompts, g_spans, g_al_self, g_al_alt, g_meta = [], [], [], [], []
    # Also 1 prompt per gap entity for extracting Donor A & Recipient B entity vectors
    ent_prompts, ent_spans = [], []
    for gi_pos, gi in enumerate(gap_items):
        pm = _t3_prompts_for_entity(gi["entity"], gi["kind"], gi["country"])
        ent_prompts.append(pm["country"]["prompt"])
        ent_spans.append(pm["country"]["span"])
        alt_c = gap_items[alt_idx_map[gi_pos]]["country"]
        pm_alt = _t3_prompts_for_entity(gi["entity"], gi["kind"], alt_c)
        for ak in t3_attrs:
            g_prompts.append(pm[ak]["prompt"])
            g_spans.append(pm[ak]["span"])
            g_al_self.append(pm[ak]["aliases"])
            g_al_alt.append(pm_alt[ak]["aliases"])
            g_meta.append({"gi_pos": gi_pos, "attr": ak, "entity": gi["entity"], "country": gi["country"], "alt_country": alt_c})

    # Unpatched baseline on B across all 5 attributes
    base_b_ev = eval_prompts_patched(model_B, tok_B, g_prompts, g_spans, g_al_self, device=device, batch_size=32)
    base_a_ev = eval_prompts_patched(model_A, tok_A, g_prompts, g_spans, g_al_self, device=device, batch_size=32)

    def _summarize_5attr(em_list: Sequence[int]) -> dict:
        out = {}
        for ak in t3_attrs:
            idxs = [i for i, m in enumerate(g_meta) if m["attr"] == ak]
            out[ak] = round(float(np.mean([em_list[i] for i in idxs])), 4)
        out["unseen_4attr_mean"] = round(float(np.mean([out[ak] for ak in ("capital", "currency", "language", "continent")])), 4)
        return out

    # Extract Donor A & Recipient B states at (17, 9) and (20, 11)
    t3_layer_pairs = [(17, 9), (20, 11)]
    st_A = extract_multi_layer_states(model_A, tok_A, ent_prompts, ent_spans, [17, 20], device)["resid"]
    st_B = extract_multi_layer_states(model_B, tok_B, ent_prompts, ent_spans, [9, 11], device)["resid"]

    R_rand = _polar_f64(rng.standard_normal((2048, 2048)))

    t3_res: Dict[str, Any] = {
        "n_screened": len(T3_ENTITIES),
        "n_A_right": int(sum(res_A["em_new"])),
        "n_B_right": int(sum(res_B["em_new"])),
        "n_gap_A_right_B_wrong": len(gap_items),
        "gap_entities_sample": gap_items[:15],
        "unpatched_A_on_gap": _summarize_5attr(base_a_ev["em_new"]),
        "unpatched_B_on_gap": _summarize_5attr(base_b_ev["em_new"]),
        "conditions": {},
    }
    print(f"  [T3 unpatched] A_on_gap={t3_res['unpatched_A_on_gap']} | B_on_gap={t3_res['unpatched_B_on_gap']}")

    # Conditions to test:
    #   1. lA17->lB9 Pilot 1 10k Procrustes (matched, diff-gap-entity, B-self-roundtrip, random-rot)
    #   2. lA17->lB9 Pilot 1 10k Fisher-Orth O(d) (matched)
    #   3. lA20->lB11 leak-free 2.5k Procrustes (matched + diff-gap-entity + random-rot)
    p1_proc = pilot1_10k_maps["procrustes_global"]
    p1_fish = pilot1_10k_maps["fisher_orth_Od_multi"]

    def _map_z(zA: np.ndarray, mp: dict) -> np.ndarray:
        return mp["mu_B"] + mp["s"] * ((zA.astype(np.float64) - mp["mu_A"]) @ mp["R"].T)

    cond_specs = [
        ("l17_9_p1_10k_procrustes_matched", 17, 9, _map_z(st_A[17], p1_proc), False),
        ("l17_9_p1_10k_fisher_orth_matched", 17, 9, _map_z(st_A[17], p1_fish), False),
        ("l17_9_ctrl_diff_gap_entity", 17, 9, _map_z(st_A[17][alt_idx_map], p1_proc), True),
        ("l17_9_ctrl_B_self_roundtrip", 17, 9, st_B[9].astype(np.float64), False),
        (
            "l17_9_ctrl_norm_matched_random_rot", 17, 9,
            p1_proc["mu_B"] + p1_proc["s"] * ((st_A[17].astype(np.float64) - p1_proc["mu_A"]) @ R_rand.T), False,
        ),
        ("l20_11_unlab2500_procrustes_matched", 20, 11, _map_z(st_A[20], resid_maps[(20, 11)]), False),
        ("l20_11_ctrl_diff_gap_entity", 20, 11, _map_z(st_A[20][alt_idx_map], resid_maps[(20, 11)]), True),
        (
            "l20_11_ctrl_norm_matched_random_rot", 20, 11,
            resid_maps[(20, 11)]["mu_B"] + resid_maps[(20, 11)]["s"] * ((st_A[20].astype(np.float64) - resid_maps[(20, 11)]["mu_A"]) @ R_rand.T), False,
        ),
    ]

    for cname, la, lb, z_hat, score_alt in cond_specs:
        z_exp = np.repeat(z_hat, len(t3_attrs), axis=0)
        ev = eval_prompts_patched(
            model_B, tok_B, g_prompts, g_spans, g_al_self,
            old_aliases_list=g_al_alt, layer_idx=lb, replace_mat=z_exp, device=device, batch_size=32,
        )
        entry = {"scored_on_orig_entity": _summarize_5attr(ev["em_new"])}
        if score_alt:
            entry["scored_on_swapped_alt_entity"] = _summarize_5attr(ev["em_old"])
        # Save a few concrete transplanted examples for matched condition
        if cname == "l17_9_p1_10k_procrustes_matched":
            exs = []
            for gi_pos in range(min(15, len(gap_items))):
                base_k = gi_pos * len(t3_attrs)
                exs.append({
                    "entity": gap_items[gi_pos]["entity"],
                    "true_country": gap_items[gi_pos]["country"],
                    "unpatched_B_country_gen": gap_items[gi_pos]["B_gen"],
                    "transplanted_gens": {t3_attrs[a]: ev["gens"][base_k + a] for a in range(len(t3_attrs))},
                    "transplanted_ok": {t3_attrs[a]: ev["em_new"][base_k + a] for a in range(len(t3_attrs))},
                })
            entry["examples"] = exs
        t3_res["conditions"][cname] = entry
        s_o = entry["scored_on_orig_entity"]
        alt_str = f" | tracks_alt={entry['scored_on_swapped_alt_entity']}" if score_alt else ""
        print(f"  [T3 {cname:36s}] orig={s_o}{alt_str}")

    return t3_res


# ======================================================================================
# 6. Fast CPU Unit-Test (Zero GPU!) & Main CLI
# ======================================================================================

def run_cpu_unit_test():
    """Verify all data integrity, derangement constraints, and float64 math kernels on CPU in ~1s."""
    print("[unit-test] Checking benchmark tables & derangement constraints...")
    assert len(LANDMARKS_T0) == 32
    assert len(T3_ENTITIES) >= 135
    for lm in LANDMARKS_T0 + PEOPLE_FALLBACK_T0:
        assert lm["country"] in COUNTRIES_INFO, f"Missing country: {lm['country']}"
    for ent, kind, c in T3_ENTITIES:
        assert c in COUNTRIES_INFO, f"Missing T3 country: {c}"
    der = assign_deranged_targets(LANDMARKS_T0, seed=SEED)
    assert len(der) == len(LANDMARKS_T0)
    by_old = defaultdict(list)
    for lm in LANDMARKS_T0:
        c_old, c_new = lm["country"], der[lm["name"]]
        assert c_old != c_new
        assert COUNTRIES_INFO[c_old]["capital"][0] != COUNTRIES_INFO[c_new]["capital"][0]
        assert COUNTRIES_INFO[c_old]["currency"][0] != COUNTRIES_INFO[c_new]["currency"][0]
        assert COUNTRIES_INFO[c_old]["language"][0] != COUNTRIES_INFO[c_new]["language"][0]
        by_old[c_old].append(c_new)
    for c_old, tgts in by_old.items():
        assert len(tgts) == len(set(tgts)), f"Duplicate targets for {c_old}: {tgts}"
    print(f"  Benchmark OK: 32 landmarks (all derangements 4-way disjoint & unique per old country), {len(T3_ENTITIES)} T3 entities!")

    print("[unit-test] Checking float64 Procrustes, Platonic socket, MVDR filter, and Gate 3...")
    rng = np.random.default_rng(42)
    N, D = 256, 64
    A = rng.standard_normal((N, D))
    Q_true = _polar_f64(rng.standard_normal((D, D)))
    B = 1.7 * (A @ Q_true.T) + 0.1 * rng.standard_normal((N, D))
    mp = fit_procrustes_f64(A, B)
    assert np.linalg.norm(mp["R"].T @ mp["R"] - np.eye(D), "fro") < 1e-12
    sock = fit_platonic_socket_f64(A, B)
    assert sock["r_corr"].shape == (D,) and sock["r_corr"][0] > 0.95

    C_bg = (B.T @ B) / N
    tgt = rng.standard_normal(D)
    hneg = rng.standard_normal((5, D))
    g = compute_mvdr_filter(tgt, hneg, C_bg)
    assert abs(np.linalg.norm(g) - 1.0) < 1e-12
    # Verify MVDR nulls hard negatives relative to target!
    assert abs(float(g @ tgt)) > 10.0 * max(abs(float(g @ h)) for h in hneg)

    # Gate 3 constant vector check (add constant feature + noise)
    H_const = rng.standard_normal((200, D))
    H_const[:, 0] = 1.0 + 0.01 * rng.standard_normal(200)
    g3 = fit_implicit_constant_gate3(H_const[:100], H_const[100:])
    assert g3["eval_std_sigma_c"] < 0.05

    # Verify Pilot 1 cached maps & unlabeled pack exist and load cleanly
    assert PILOT1_MAPS_PATH.exists(), f"Missing {PILOT1_MAPS_PATH}"
    assert PILOT1_UNLAB_PATH.exists(), f"Missing {PILOT1_UNLAB_PATH}"
    p1m = torch.load(PILOT1_MAPS_PATH, map_location="cpu", weights_only=False)
    assert "procrustes_global" in p1m and "fisher_orth_Od_multi" in p1m
    print(f"  Math & cache checks 100% PASSED! (MVDR nulling verified, Gate 3 sigma_c={g3['eval_std_sigma_c']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Consensus 4-Test Grafting Falsification Pilot (T0-T3)")
    parser.add_argument("--tests", default="t0,t1,t2,t3", help="Comma-separated subset of t0,t1,t2,t3")
    parser.add_argument("--n-cal", type=int, default=2500, help="Leak-free unlabeled calibration spans for swept layers")
    parser.add_argument("--unit-test", action="store_true", help="CPU-only math/data check (zero GPU)")
    args = parser.parse_args()

    if args.unit_test:
        run_cpu_unit_test()
        return

    t_start = time.time()
    wanted = {x.strip().lower() for x in args.tests.split(",") if x.strip()}
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[graft-falsify] tests={sorted(wanted)} device={device} n_cal={args.n_cal}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(CACHE_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dir_A = Path(manifest["models"][DONOR_KEY]["local_path"])
    dir_B = Path(manifest["models"][RECIP_KEY]["local_path"])

    # Load both models in float32 (~12GB + ~6.8GB = ~19GB out of 48GB unified RAM — clean & fast!)
    print(f"[graft-falsify] Loading {DONOR_KEY} and {RECIP_KEY} in float32 on {device}...")
    model_A, tok_A = load_model_f32(dir_A, device)
    model_B, tok_B = load_model_f32(dir_B, device)

    results: Dict[str, Any] = {
        "meta": {
            "donor": DONOR_KEY,
            "recipient": RECIP_KEY,
            "device": str(device),
            "tests_run": sorted(wanted),
            "swept_layer_pairs": SWEPT_LAYER_PAIRS,
            "seed": SEED,
        }
    }

    # Build leak-free multi-layer calibration maps & post-RMSNorm sockets ONCE
    resid_maps, mlp_sockets, cal_extra = build_leak_free_calibration_maps(
        model_A, tok_A, model_B, tok_B, device, n_cal=args.n_cal
    )
    results["calibration_meta"] = cal_extra["cal_meta"]

    passing_items = LANDMARKS_T0
    if "t0" in wanted or "t1" in wanted or "t2" in wanted:
        passing_items, t0_res = run_t0_competence_filter(model_A, tok_A, model_B, tok_B, device)
        results["T0_competence_filter"] = t0_res

    if "t1" in wanted:
        results["T1_forced_gate_audit"] = run_t1_forced_gate_audit(
            passing_items, model_A, tok_A, model_B, tok_B, resid_maps, device
        )

    if "t2" in wanted:
        results["T2_key_selectivity_and_gate3"] = run_t2_key_selectivity_and_gate3(
            passing_items, dataset, model_A, tok_A, model_B, tok_B, mlp_sockets, device
        )

    if "t3" in wanted:
        results["T3_knowledge_gap_transplant"] = run_t3_knowledge_gap_transplant(
            model_A, tok_A, model_B, tok_B, resid_maps, cal_extra["pilot1_10k"], device
        )

    del model_A, model_B
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    results["meta"]["elapsed_sec"] = round(time.time() - t_start, 2)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[graft-falsify] COMPLETE! Saved to {OUT_PATH} in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
