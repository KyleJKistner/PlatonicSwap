#!/usr/bin/env python3
"""
pilot_graft_t3_gap.py — Standalone Fable T3 Knowledge-Gap Noun Transplant (~2 min MPS)
======================================================================================

Screens ~200 genuinely obscure/long-tail real-world entities (regional towns, minor rivers,
islands, mountains, regional universities, plus dataset.json A-right/B-wrong items) for
GENUINE knowledge gaps where Donor A (qwen2.5-3b) knows the country (and/or dependent
attributes) while Recipient B (smollm2-1.7b) fails!

Transplants Donor A's single last_subj residual vector into Recipient B via Pilot 1's
full-rank N=10,000 zero-QA maps (10k_procrustes_global & 10k_fisher_orth_Od_multi at
lA=17 -> lB=9, plus 2.5k leak-free map at lA=20 -> lB=11), testing whether Recipient B
now answers `country` AND 4 unseen dependent attributes (`capital`, `currency`, `language`,
`continent`) — with all 3 mandatory controls:
  1. Different gap entity e' from A (B's answers must track e' and drop on e!)
  2. B's own self round-trip (must stay wrong!)
  3. Norm-matched random rotation R_rand (must stay wrong!)

Usage:
  python pilot_graft_t3_gap.py              # Full MPS run (~2 min -> results_graft_t3_gap.json)
  python pilot_graft_t3_gap.py --unit-test  # Fast CPU-only check (zero GPU!)
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from extract_and_baseline import CACHE_DIR, DATASET_PATH, LAB_DIR, SEED
from pilot_graft_falsify import (
    COUNTRIES_INFO,
    DONOR_KEY,
    PILOT1_MAPS_PATH,
    RECIP_KEY,
    _polar_f64,
    eval_prompts_patched,
    extract_multi_layer_states,
    load_model_f32,
)

OUT_PATH = LAB_DIR / "results_graft_t3_gap.json"

# Extend COUNTRIES_INFO with a few extra countries used in obscure pool
EXTRA_COUNTRIES: Dict[str, Dict[str, Any]] = {
    "Pakistan": {
        "aliases": ["Pakistan"],
        "capital": ("Islamabad", ["Islamabad"]),
        "currency": ("rupee", ["rupee", "Pakistani rupee"]),
        "language": ("Urdu", ["Urdu"]),
        "continent": ("Asia", ["Asia"]),
    },
    "Ecuador": {
        "aliases": ["Ecuador"],
        "capital": ("Quito", ["Quito"]),
        "currency": ("dollar", ["dollar", "US dollar", "U.S. dollar"]),
        "language": ("Spanish", ["Spanish"]),
        "continent": ("South America", ["South America"]),
    },
    "Nepal": {
        "aliases": ["Nepal"],
        "capital": ("Kathmandu", ["Kathmandu"]),
        "currency": ("rupee", ["rupee", "Nepalese rupee"]),
        "language": ("Nepali", ["Nepali"]),
        "continent": ("Asia", ["Asia"]),
    },
    "Tanzania": {
        "aliases": ["Tanzania"],
        "capital": ("Dodoma", ["Dodoma", "Dar es Salaam"]),
        "currency": ("shilling", ["shilling", "Tanzanian shilling"]),
        "language": ("Swahili", ["Swahili"]),
        "continent": ("Africa", ["Africa"]),
    },
    "Iceland": {
        "aliases": ["Iceland"],
        "capital": ("Reykjavik", ["Reykjavik", "Reykjavík"]),
        "currency": ("krona", ["krona", "króna", "Icelandic krona", "Icelandic króna"]),
        "language": ("Icelandic", ["Icelandic"]),
        "continent": ("Europe", ["Europe"]),
    },
    "Latvia": {
        "aliases": ["Latvia"],
        "capital": ("Riga", ["Riga"]),
        "currency": ("euro", ["euro", "Euro"]),
        "language": ("Latvian", ["Latvian"]),
        "continent": ("Europe", ["Europe"]),
    },
    "Lithuania": {
        "aliases": ["Lithuania"],
        "capital": ("Vilnius", ["Vilnius"]),
        "currency": ("euro", ["euro", "Euro"]),
        "language": ("Lithuanian", ["Lithuanian"]),
        "continent": ("Europe", ["Europe"]),
    },
    "Serbia": {
        "aliases": ["Serbia"],
        "capital": ("Belgrade", ["Belgrade"]),
        "currency": ("dinar", ["dinar", "Serbian dinar"]),
        "language": ("Serbian", ["Serbian"]),
        "continent": ("Europe", ["Europe"]),
    },
    "Tunisia": {
        "aliases": ["Tunisia"],
        "capital": ("Tunis", ["Tunis"]),
        "currency": ("dinar", ["dinar", "Tunisian dinar"]),
        "language": ("Arabic", ["Arabic"]),
        "continent": ("Africa", ["Africa"]),
    },
    "Nigeria": {
        "aliases": ["Nigeria"],
        "capital": ("Abuja", ["Abuja"]),
        "currency": ("naira", ["naira", "Nigerian naira"]),
        "language": ("English", ["English"]),
        "continent": ("Africa", ["Africa"]),
    },
    "Ethiopia": {
        "aliases": ["Ethiopia"],
        "capital": ("Addis Ababa", ["Addis Ababa"]),
        "currency": ("birr", ["birr", "Ethiopian birr"]),
        "language": ("Amharic", ["Amharic"]),
        "continent": ("Africa", ["Africa"]),
    },
    "Bolivia": {
        "aliases": ["Bolivia"],
        "capital": ("Sucre", ["Sucre", "La Paz"]),
        "currency": ("boliviano", ["boliviano", "Bolivian boliviano"]),
        "language": ("Spanish", ["Spanish"]),
        "continent": ("South America", ["South America"]),
    },
    "Uruguay": {
        "aliases": ["Uruguay"],
        "capital": ("Montevideo", ["Montevideo"]),
        "currency": ("peso", ["peso", "Uruguayan peso"]),
        "language": ("Spanish", ["Spanish"]),
        "continent": ("South America", ["South America"]),
    },
    "Paraguay": {
        "aliases": ["Paraguay"],
        "capital": ("Asuncion", ["Asuncion", "Asunción"]),
        "currency": ("guarani", ["guarani", "guaraní", "Paraguayan guarani"]),
        "language": ("Spanish", ["Spanish", "Guarani"]),
        "continent": ("South America", ["South America"]),
    },
}

ALL_COUNTRIES: Dict[str, Dict[str, Any]] = {**COUNTRIES_INFO, **EXTRA_COUNTRIES}

# ~190 genuinely obscure/long-tail real-world entities across towns, rivers, islands, mountains, universities, historical/cultural items
OBSCURE_ENTITIES: List[Tuple[str, str, str]] = [
    # Small/regional towns worldwide
    ("Klagenfurt", "city", "Austria"), ("Villach", "city", "Austria"), ("Bregenz", "city", "Austria"),
    ("Stara Zagora", "city", "Bulgaria"), ("Ruse", "city", "Bulgaria"), ("Burgas", "city", "Bulgaria"),
    ("Parnu", "city", "Estonia"), ("Narva", "city", "Estonia"),
    ("Daugavpils", "city", "Latvia"), ("Liepaja", "city", "Latvia"),
    ("Kaunas", "city", "Lithuania"), ("Klaipeda", "city", "Lithuania"),
    ("Olomouc", "city", "Czech Republic"), ("Plzen", "city", "Czech Republic"), ("Liberec", "city", "Czech Republic"),
    ("Evora", "city", "Portugal"), ("Aveiro", "city", "Portugal"), ("Setubal", "city", "Portugal"),
    ("Hasselt", "city", "Belgium"), ("Namur", "city", "Belgium"), ("Mons", "city", "Belgium"),
    ("Aalborg", "city", "Denmark"), ("Esbjerg", "city", "Denmark"), ("Randers", "city", "Denmark"),
    ("Jyvaskyla", "city", "Finland"), ("Kuopio", "city", "Finland"), ("Vaasa", "city", "Finland"),
    ("Celje", "city", "Slovenia"), ("Koper", "city", "Slovenia"),
    ("Zadar", "city", "Croatia"), ("Osijek", "city", "Croatia"), ("Pula", "city", "Croatia"),
    ("Nis", "city", "Serbia"), ("Novi Sad", "city", "Serbia"), ("Kragujevac", "city", "Serbia"),
    ("Iasi", "city", "Romania"), ("Oradea", "city", "Romania"), ("Sibiu", "city", "Romania"), ("Craiova", "city", "Romania"),
    ("Gyor", "city", "Hungary"), ("Miskolc", "city", "Hungary"), ("Eger", "city", "Hungary"),
    ("Torun", "city", "Poland"), ("Bydgoszcz", "city", "Poland"), ("Lublin", "city", "Poland"), ("Rzeszow", "city", "Poland"), ("Szczecin", "city", "Poland"),
    ("Umea", "city", "Sweden"), ("Linkoping", "city", "Sweden"), ("Orebro", "city", "Sweden"), ("Lund", "city", "Sweden"),
    ("Tromso", "city", "Norway"), ("Kristiansand", "city", "Norway"), ("Alesund", "city", "Norway"),
    ("Akureyri", "city", "Iceland"),
    ("Waterford", "city", "Ireland"), ("Kilkenny", "city", "Ireland"), ("Sligo", "city", "Ireland"),
    ("Presov", "city", "Slovakia"), ("Nitra", "city", "Slovakia"), ("Zilina", "city", "Slovakia"),
    ("Ioannina", "city", "Greece"), ("Larissa", "city", "Greece"), ("Volos", "city", "Greece"), ("Kavala", "city", "Greece"),
    ("Nijmegen", "city", "Netherlands"), ("Arnhem", "city", "Netherlands"), ("Maastricht", "city", "Netherlands"), ("Breda", "city", "Netherlands"),
    ("Fribourg", "city", "Switzerland"), ("Lugano", "city", "Switzerland"), ("Neuchatel", "city", "Switzerland"),
    ("Vigo", "city", "Spain"), ("Murcia", "city", "Spain"), ("Oviedo", "city", "Spain"), ("Pamplona", "city", "Spain"),
    ("Cagliari", "city", "Italy"), ("Bari", "city", "Italy"), ("Catania", "city", "Italy"), ("Perugia", "city", "Italy"), ("Trieste", "city", "Italy"),
    ("Rennes", "city", "France"), ("Reims", "city", "France"), ("Dijon", "city", "France"), ("Angers", "city", "France"), ("Clermont-Ferrand", "city", "France"),
    ("Augsburg", "city", "Germany"), ("Freiburg", "city", "Germany"), ("Kiel", "city", "Germany"), ("Rostock", "city", "Germany"), ("Regensburg", "city", "Germany"),
    ("Exeter", "city", "United Kingdom"), ("Norwich", "city", "United Kingdom"), ("Inverness", "city", "United Kingdom"), ("Swansea", "city", "United Kingdom"),
    ("Tauranga", "city", "New Zealand"), ("Napier", "city", "New Zealand"), ("Invercargill", "city", "New Zealand"),
    ("Launceston", "city", "Australia"), ("Townsville", "city", "Australia"), ("Ballarat", "city", "Australia"),
    ("Matsuyama", "city", "Japan"), ("Kumamoto", "city", "Japan"), ("Kagoshima", "city", "Japan"), ("Akita", "city", "Japan"), ("Niigata", "city", "Japan"), ("Takamatsu", "city", "Japan"),
    ("Ulsan", "city", "South Korea"), ("Cheongju", "city", "South Korea"), ("Jeonju", "city", "South Korea"), ("Pohang", "city", "South Korea"),
    ("Khon Kaen", "city", "Thailand"), ("Hat Yai", "city", "Thailand"), ("Udon Thani", "city", "Thailand"),
    ("Ipoh", "city", "Malaysia"), ("Kuching", "city", "Malaysia"), ("Kota Kinabalu", "city", "Malaysia"),
    ("Makassar", "city", "Indonesia"), ("Semarang", "city", "Indonesia"), ("Palembang", "city", "Indonesia"), ("Malang", "city", "Indonesia"),
    ("Iloilo", "city", "Philippines"), ("Bacolod", "city", "Philippines"), ("Zamboanga", "city", "Philippines"),
    ("Antofagasta", "city", "Chile"), ("Temuco", "city", "Chile"), ("La Serena", "city", "Chile"),
    ("Trujillo", "city", "Peru"), ("Piura", "city", "Peru"), ("Iquitos", "city", "Peru"),
    ("Bucaramanga", "city", "Colombia"), ("Pereira", "city", "Colombia"), ("Manizales", "city", "Colombia"),
    ("Tucuman", "city", "Argentina"), ("Salta", "city", "Argentina"), ("Neuquen", "city", "Argentina"),
    ("Cochabamba", "city", "Bolivia"), ("Oruro", "city", "Bolivia"),
    ("Salto", "city", "Uruguay"), ("Paysandu", "city", "Uruguay"),
    ("Encarnacion", "city", "Paraguay"),
    ("Belem", "city", "Brazil"), ("Goiania", "city", "Brazil"), ("Florianopolis", "city", "Brazil"), ("Campinas", "city", "Brazil"),
    ("Merida", "city", "Mexico"), ("Oaxaca", "city", "Mexico"), ("Queretaro", "city", "Mexico"), ("Veracruz", "city", "Mexico"),
    ("Cuenca", "city", "Ecuador"), ("Ambato", "city", "Ecuador"),
    ("Moncton", "city", "Canada"), ("Sudbury", "city", "Canada"), ("Kelowna", "city", "Canada"),
    ("Meknes", "city", "Morocco"), ("Oujda", "city", "Morocco"), ("Agadir", "city", "Morocco"),
    ("Sfax", "city", "Tunisia"), ("Sousse", "city", "Tunisia"),
    ("Tanta", "city", "Egypt"), ("Mansoura", "city", "Egypt"), ("Suez", "city", "Egypt"),
    ("Eldoret", "city", "Kenya"), ("Nakuru", "city", "Kenya"),
    ("Arusha", "city", "Tanzania"), ("Mwanza", "city", "Tanzania"),
    ("Gondar", "city", "Ethiopia"), ("Mekelle", "city", "Ethiopia"),
    ("Kano", "city", "Nigeria"), ("Ibadan", "city", "Nigeria"), ("Kaduna", "city", "Nigeria"),
    ("Tamale", "city", "Ghana"), ("Takoradi", "city", "Ghana"),
    ("Polokwane", "city", "South Africa"), ("Kimberley", "city", "South Africa"), ("Nelspruit", "city", "South Africa"),
    ("Kerman", "city", "Iran"), ("Yazd", "city", "Iran"), ("Ahvaz", "city", "Iran"), ("Rasht", "city", "Iran"),
    ("Multan", "city", "Pakistan"), ("Peshawar", "city", "Pakistan"), ("Quetta", "city", "Pakistan"),
    ("Pokhara", "city", "Nepal"), ("Biratnagar", "city", "Nepal"),
    ("Namangan", "city", "Uzbekistan"), ("Andijan", "city", "Uzbekistan"), ("Nukus", "city", "Uzbekistan"),
    ("Shymkent", "city", "Kazakhstan"), ("Karaganda", "city", "Kazakhstan"), ("Aktobe", "city", "Kazakhstan"),
    ("Ganja", "city", "Azerbaijan"), ("Sumqayit", "city", "Azerbaijan"),
    ("Batumi", "city", "Georgia"), ("Kutaisi", "city", "Georgia"),
    ("Gyumri", "city", "Armenia"), ("Vanadzor", "city", "Armenia"),
    ("Konya", "city", "Turkey"), ("Gaziantep", "city", "Turkey"), ("Kayseri", "city", "Turkey"), ("Trabzon", "city", "Turkey"), ("Samsun", "city", "Turkey"),
    ("Beersheba", "city", "Israel"), ("Netanya", "city", "Israel"), ("Ashdod", "city", "Israel"),
    ("Can Tho", "city", "Vietnam"), ("Nha Trang", "city", "Vietnam"), ("Haiphong", "city", "Vietnam"),
    # Minor rivers, mountains, islands, regional landmarks
    ("Tisza River", "river", "Hungary"), ("Weser River", "river", "Germany"), ("Ebro River", "river", "Spain"),
    ("Adige River", "river", "Italy"), ("Glomma River", "river", "Norway"), ("Kemijoki River", "river", "Finland"),
    ("Ouse River", "river", "United Kingdom"), ("Shannon River", "river", "Ireland"), ("Sava River", "river", "Croatia"),
    ("Gotland", "island", "Sweden"), ("Bornholm", "island", "Denmark"), ("Fyn", "island", "Denmark"),
    ("Madeira", "island", "Portugal"), ("Shikoku", "island", "Japan"), ("Sulawesi", "island", "Indonesia"),
    ("Mindanao", "island", "Philippines"), ("Luzon", "island", "Philippines"), ("Tasmania", "island", "Australia"),
    ("Mount Kinabalu", "mountain", "Malaysia"), ("Mount Toubkal", "mountain", "Morocco"),
    ("Mount Damavand", "mountain", "Iran"), ("Mount Ararat", "mountain", "Turkey"),
    ("Mount Kosciuszko", "mountain", "Australia"), ("Mount Teide", "mountain", "Spain"),
    ("Mount Merapi", "mountain", "Indonesia"), ("Mount Hallasan", "mountain", "South Korea"),
    ("Huascaran", "mountain", "Peru"), ("Cotopaxi", "mountain", "Ecuador"), ("Chimborazo", "mountain", "Ecuador"),
    # Regional universities & airlines/companies
    ("University of Debrecen", "university", "Hungary"),
    ("University of Szeged", "university", "Hungary"),
    ("University of Turku", "university", "Finland"),
    ("University of Bergen", "university", "Norway"),
    ("University of Ljubljana", "university", "Slovenia"),
    ("University of Zagreb", "university", "Croatia"),
    ("University of Porto", "university", "Portugal"),
    ("University of Groningen", "university", "Netherlands"),
    ("University of Padua", "university", "Italy"),
    ("University of Aberdeen", "university", "United Kingdom"),
    ("University of Canterbury", "university", "New Zealand"),
    ("University of Tasmania", "university", "Australia"),
    ("Chulalongkorn University", "university", "Thailand"),
    ("Gadjah Mada University", "university", "Indonesia"),
    ("Ateneo de Manila University", "university", "Philippines"),
    ("Yonsei University", "university", "South Korea"),
    ("Hokkaido University", "university", "Japan"),
    ("Nagoya University", "university", "Japan"),
]


def _t3_prompts(ent: str, kind: str, country: str) -> Dict[str, dict]:
    c_info = ALL_COUNTRIES[country]
    if kind == "city":
        pref = f"The city of {ent}"
        c_prompt = f"{pref} is located in the country of"
    elif kind == "university":
        pref = f"The {ent}" if ent.startswith("University") else f"The university {ent}"
        c_prompt = f"{pref} is located in the country of"
    elif kind == "river":
        pref = f"The {ent}"
        c_prompt = f"{pref} flows primarily through the country of"
    elif kind == "island":
        pref = f"The island of {ent}"
        c_prompt = f"{pref} belongs to the country of"
    elif kind == "mountain":
        pref = f"{ent}" if ent.startswith("Mount ") else f"The mountain {ent}"
        c_prompt = f"{pref} is located in the country of"
    elif kind == "airline":
        pref = f"The airline {ent}"
        c_prompt = f"{pref} is based in the country of"
    elif kind == "company":
        pref = f"The company {ent}"
        c_prompt = f"{pref} is headquartered in the country of"
    else:
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


def collect_dataset_gap_candidates() -> List[Tuple[str, str, str, str]]:
    """Pull A-right, B-wrong country-related items from dataset.json (excluding cities where B only gave province if possible)."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        ds = json.load(f)["facts"]
    with open(CACHE_DIR / f"{DONOR_KEY}_eval.json", "r", encoding="utf-8") as f:
        eA = json.load(f)["facts"]
    with open(CACHE_DIR / f"{RECIP_KEY}_eval.json", "r", encoding="utf-8") as f:
        eB = json.load(f)["facts"]

    rel_to_kind = {
        "airline_country": "airline",
        "company_country": "company",
        "city_country": "city",
        "university_country": "university",
        "landmark_country": "landmark",
    }
    seen = {e.lower() for e, _, _ in OBSCURE_ENTITIES}
    out = []
    for f_it, a, b in zip(ds, eA, eB):
        rel = f_it["relation"]
        if rel not in rel_to_kind:
            continue
        ans = f_it["answer"]
        if ans not in ALL_COUNTRIES:
            continue
        if a["direct_correct"] and not b["direct_correct"]:
            subj = f_it["subject"]
            if subj.lower() not in seen:
                seen.add(subj.lower())
                out.append((subj, rel_to_kind[rel], ans, "dataset_gap"))
    return out


def run_unit_test():
    for ent, kind, c in OBSCURE_ENTITIES:
        assert c in ALL_COUNTRIES, f"Missing country {c} for {ent}"
        pm = _t3_prompts(ent, kind, c)
        assert len(pm) == 5
    ds_gaps = collect_dataset_gap_candidates()
    assert PILOT1_MAPS_PATH.exists()
    print(f"[unit-test] OK! {len(OBSCURE_ENTITIES)} obscure entities + {len(ds_gaps)} dataset_gap candidates = {len(OBSCURE_ENTITIES) + len(ds_gaps)} total!")


def main():
    parser = argparse.ArgumentParser(description="Standalone Fable T3 Knowledge-Gap Noun Transplant")
    parser.add_argument("--unit-test", action="store_true")
    args = parser.parse_args()
    if args.unit_test:
        run_unit_test()
        return

    t0 = time.time()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    with open(CACHE_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    pool: List[Tuple[str, str, str, str]] = [
        (e, k, c, "obscure_longtail") for e, k, c in OBSCURE_ENTITIES
    ] + collect_dataset_gap_candidates()
    print(f"[T3-gap] Screening {len(pool)} entities on {DONOR_KEY} (A) & {RECIP_KEY} (B) on {device}...")

    model_A, tok_A = load_model_f32(Path(manifest["models"][DONOR_KEY]["local_path"]), device)
    model_B, tok_B = load_model_f32(Path(manifest["models"][RECIP_KEY]["local_path"]), device)

    # Screen country prompts on A and B
    scr_p, scr_sp, scr_al = [], [], []
    for ent, kind, c, _ in pool:
        pm = _t3_prompts(ent, kind, c)["country"]
        scr_p.append(pm["prompt"])
        scr_sp.append(pm["span"])
        scr_al.append(pm["aliases"])

    res_A = eval_prompts_patched(model_A, tok_A, scr_p, scr_sp, scr_al, device=device, batch_size=32)
    res_B = eval_prompts_patched(model_B, tok_B, scr_p, scr_sp, scr_al, device=device, batch_size=32)

    gap_items = []
    for i, (ent, kind, c, src) in enumerate(pool):
        if res_A["em_new"][i] == 1 and res_B["em_new"][i] == 0:
            gap_items.append({
                "entity": ent, "kind": kind, "country": c, "source": src,
                "A_gen": res_A["gens"][i], "B_gen": res_B["gens"][i],
            })

    n_obs_gap = sum(1 for x in gap_items if x["source"] == "obscure_longtail")
    n_ds_gap = sum(1 for x in gap_items if x["source"] == "dataset_gap")
    print(
        f"[T3-gap] A_right={sum(res_A['em_new'])}/{len(pool)}, B_right={sum(res_B['em_new'])}/{len(pool)} | "
        f"GAP (A-right, B-wrong)={len(gap_items)} (obscure={n_obs_gap}, dataset_gap={n_ds_gap})"
    )

    if len(gap_items) < 4:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"n_screened": len(pool), "n_gap": len(gap_items), "gap_items": gap_items}, f, indent=2)
        return

    # Assign different gap entity e' (different country/capital/currency) for Control 1
    rng = np.random.default_rng(SEED + 31337)
    alt_idx_map = []
    for i, gi in enumerate(gap_items):
        ci = ALL_COUNTRIES[gi["country"]]
        cands = [
            j for j, gj in enumerate(gap_items)
            if gj["country"] != gi["country"]
            and ALL_COUNTRIES[gj["country"]]["capital"][0] != ci["capital"][0]
            and ALL_COUNTRIES[gj["country"]]["currency"][0] != ci["currency"][0]
        ]
        if not cands:
            cands = [j for j, gj in enumerate(gap_items) if gj["country"] != gi["country"]]
        alt_idx_map.append(int(cands[int(rng.integers(0, len(cands)))]) if cands else (i + 1) % len(gap_items))

    t3_attrs = ["country", "capital", "currency", "language", "continent"]
    g_prompts, g_spans, g_al_self, g_al_alt, g_meta = [], [], [], [], []
    ent_prompts, ent_spans = [], []
    for gi_pos, gi in enumerate(gap_items):
        pm = _t3_prompts(gi["entity"], gi["kind"], gi["country"])
        ent_prompts.append(pm["country"]["prompt"])
        ent_spans.append(pm["country"]["span"])
        alt_c = gap_items[alt_idx_map[gi_pos]]["country"]
        pm_alt = _t3_prompts(gi["entity"], gi["kind"], alt_c)
        for ak in t3_attrs:
            g_prompts.append(pm[ak]["prompt"])
            g_spans.append(pm[ak]["span"])
            g_al_self.append(pm[ak]["aliases"])
            g_al_alt.append(pm_alt[ak]["aliases"])
            g_meta.append({
                "gi_pos": gi_pos, "attr": ak, "entity": gi["entity"],
                "country": gi["country"], "alt_country": alt_c, "source": gi["source"],
            })

    base_a_ev = eval_prompts_patched(model_A, tok_A, g_prompts, g_spans, g_al_self, device=device, batch_size=32)
    base_b_ev = eval_prompts_patched(model_B, tok_B, g_prompts, g_spans, g_al_self, device=device, batch_size=32)

    # Identify STRICT attribute-level gaps (B wrong unpatched on that specific attribute AND A right!)
    strict_gap_mask = [
        int(base_a_ev["em_new"][i] == 1 and base_b_ev["em_new"][i] == 0) for i in range(len(g_meta))
    ]
    b_wrong_mask = [int(base_b_ev["em_new"][i] == 0) for i in range(len(g_meta))]

    def _summarize(em_list: Sequence[int], subset_mask: Optional[Sequence[int]] = None, source_filter: Optional[str] = None) -> dict:
        out = {}
        for ak in t3_attrs:
            idxs = [
                i for i, m in enumerate(g_meta)
                if m["attr"] == ak
                and (subset_mask is None or subset_mask[i] == 1)
                and (source_filter is None or m["source"] == source_filter)
            ]
            out[ak] = {"acc": round(float(np.mean([em_list[i] for i in idxs])), 4) if idxs else None, "n": len(idxs)}
        vals4 = [out[ak]["acc"] for ak in ("capital", "currency", "language", "continent") if out[ak]["acc"] is not None]
        out["unseen_4attr_mean"] = round(float(np.mean(vals4)), 4) if vals4 else None
        return out

    # Extract Donor A & Recipient B states at (17, 9)
    st_A = extract_multi_layer_states(model_A, tok_A, ent_prompts, ent_spans, [17], device)["resid"][17]
    st_B = extract_multi_layer_states(model_B, tok_B, ent_prompts, ent_spans, [9], device)["resid"][9]

    p1_maps = torch.load(PILOT1_MAPS_PATH, map_location="cpu", weights_only=False)
    p1_proc = p1_maps["procrustes_global"]
    p1_fish = p1_maps["fisher_orth_Od_multi"]
    R_rand = _polar_f64(rng.standard_normal((2048, 2048)))

    def _map_z(zA: np.ndarray, mp: dict) -> np.ndarray:
        return mp["mu_B"] + mp["s"] * ((zA.astype(np.float64) - mp["mu_A"]) @ mp["R"].T)

    conds = [
        ("p1_10k_procrustes_matched", _map_z(st_A, p1_proc), False),
        ("p1_10k_fisher_orth_matched", _map_z(st_A, p1_fish), False),
        ("ctrl_diff_gap_entity_from_A", _map_z(st_A[alt_idx_map], p1_proc), True),
        ("ctrl_B_self_roundtrip", st_B.astype(np.float64), False),
        (
            "ctrl_norm_matched_random_rot",
            p1_proc["mu_B"] + p1_proc["s"] * ((st_A.astype(np.float64) - p1_proc["mu_A"]) @ R_rand.T), False,
        ),
    ]

    results: Dict[str, Any] = {
        "meta": {
            "donor": DONOR_KEY, "recipient": RECIP_KEY, "layers": [17, 9],
            "n_screened": len(pool), "n_A_right": int(sum(res_A["em_new"])),
            "n_B_right": int(sum(res_B["em_new"])),
            "n_gap_total": len(gap_items), "n_gap_obscure": n_obs_gap, "n_gap_dataset": n_ds_gap,
        },
        "unpatched_A_all_gap": _summarize(base_a_ev["em_new"]),
        "unpatched_B_all_gap": _summarize(base_b_ev["em_new"]),
        "unpatched_B_obscure_only": _summarize(base_b_ev["em_new"], source_filter="obscure_longtail"),
        "conditions": {},
        "gap_items": gap_items,
    }
    print(f"  [unpatched] A={results['unpatched_A_all_gap']['country']} 4attr={results['unpatched_A_all_gap']['unseen_4attr_mean']} | B={results['unpatched_B_all_gap']['country']} 4attr={results['unpatched_B_all_gap']['unseen_4attr_mean']}")

    for cname, z_hat, score_alt in conds:
        z_exp = np.repeat(z_hat, len(t3_attrs), axis=0)
        ev = eval_prompts_patched(
            model_B, tok_B, g_prompts, g_spans, g_al_self,
            old_aliases_list=g_al_alt, layer_idx=9, replace_mat=z_exp, device=device, batch_size=32,
        )
        entry = {
            "all_gap_entities": _summarize(ev["em_new"]),
            "obscure_only": _summarize(ev["em_new"], source_filter="obscure_longtail"),
            "strict_A_right_B_wrong_per_attr": _summarize(ev["em_new"], subset_mask=strict_gap_mask),
            "all_B_wrong_per_attr": _summarize(ev["em_new"], subset_mask=b_wrong_mask),
        }
        if score_alt:
            entry["tracks_swapped_alt_entity"] = _summarize(ev["em_old"])
        if cname == "p1_10k_procrustes_matched":
            exs = []
            for gi_pos in range(min(20, len(gap_items))):
                bk = gi_pos * len(t3_attrs)
                exs.append({
                    "entity": gap_items[gi_pos]["entity"],
                    "source": gap_items[gi_pos]["source"],
                    "country": gap_items[gi_pos]["country"],
                    "B_unpatched_country_gen": gap_items[gi_pos]["B_gen"],
                    "transplanted_ok": {t3_attrs[a]: ev["em_new"][bk + a] for a in range(len(t3_attrs))},
                    "transplanted_gens": {t3_attrs[a]: ev["gens"][bk + a] for a in range(len(t3_attrs))},
                })
            entry["examples"] = exs
        results["conditions"][cname] = entry
        all_g = entry["all_gap_entities"]
        obs_g = entry["obscure_only"]
        str_g = entry["strict_A_right_B_wrong_per_attr"]
        alt_s = f" | alt_ctry={entry['tracks_swapped_alt_entity']['country']['acc']} alt_4attr={entry['tracks_swapped_alt_entity']['unseen_4attr_mean']}" if score_alt else ""
        print(
            f"  [{cname:30s}] ALL: ctry={all_g['country']['acc']} 4attr={all_g['unseen_4attr_mean']} | "
            f"OBSCURE: ctry={obs_g['country']['acc']} 4attr={obs_g['unseen_4attr_mean']} | "
            f"STRICT_ATTR_GAP: ctry={str_g['country']['acc']}(n={str_g['country']['n']}) 4attr={str_g['unseen_4attr_mean']}{alt_s}"
        )

    del model_A, model_B
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    results["meta"]["elapsed_sec"] = round(time.time() - t0, 2)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[T3-gap] COMPLETE in {time.time() - t0:.1f}s -> {OUT_PATH}")


if __name__ == "__main__":
    main()
