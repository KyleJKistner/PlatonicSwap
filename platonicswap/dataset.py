#!/usr/bin/env python3
"""
Factual recall benchmark dataset generator for Plato's Key experiments.

Produces:
  - dataset.json: ~950+ clean, unambiguous single-hop factual recall items across 15 diverse relations
    with exact subject span decomposition (prefix, subject, suffix), paraphrase prompts,
    same-relation 3-shot ICL prompts, globally subject-disjoint 60/20/20 train/val/test splits,
    and Leave-One-Relation-Out (LORO) fold metadata.
  - dataset_2hop.json: ~140 clean 2-hop latent bridge items (entity -> implicit bridge -> final answer)
    with explicit bridge entity metadata for latent-thought transfer experiments.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

SEED = 20261125


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


RELATION_TEMPLATES: Dict[str, Dict[str, str]] = {
    "country_capital": {
        "prefix": "The capital of ",
        "suffix": " is",
        "para_prefix": "The capital city of ",
        "para_suffix": " is",
    },
    "country_continent": {
        "prefix": "The country of ",
        "suffix": " is located on the continent of",
        "para_prefix": "Geographically, ",
        "para_suffix": " belongs to the continent of",
    },
    "country_currency": {
        "prefix": "The official currency of ",
        "suffix": " is the",
        "para_prefix": "The national currency used in ",
        "para_suffix": " is the",
    },
    "country_language": {
        "prefix": "The primary official language of ",
        "suffix": " is",
        "para_prefix": "The main language spoken in ",
        "para_suffix": " is",
    },
    "us_state_capital": {
        "prefix": "The state capital of ",
        "suffix": " is",
        "para_prefix": "The capital city of the U.S. state of ",
        "para_suffix": " is",
    },
    "city_country": {
        "prefix": "The city of ",
        "suffix": " is in the country of",
        "para_prefix": "The major city ",
        "para_suffix": " is located in the country of",
    },
    "landmark_country": {
        "prefix": "The landmark ",
        "suffix": " is in the country of",
        "para_prefix": "Tourists visit ",
        "para_suffix": " in the country of",
    },
    "element_symbol": {
        "prefix": "On the periodic table, the symbol for ",
        "suffix": " is",
        "para_prefix": "In chemistry, the chemical symbol for ",
        "para_suffix": " is",
    },
    "company_country": {
        "prefix": "The company ",
        "suffix": " is headquartered in the country of",
        "para_prefix": "The corporation ",
        "para_suffix": " is based in the country of",
    },
    "company_founder": {
        "prefix": "The company ",
        "suffix": " was founded by",
        "para_prefix": "The founder of ",
        "para_suffix": " is",
    },
    "university_country": {
        "prefix": "Located in a major nation, ",
        "suffix": " is in the country of",
        "para_prefix": "The academic institution ",
        "para_suffix": " is located in the country of",
    },
    "airline_country": {
        "prefix": "The airline ",
        "suffix": " is based in the country of",
        "para_prefix": "The flag carrier ",
        "para_suffix": " is from the country of",
    },
    "car_brand_country": {
        "prefix": "The automaker ",
        "suffix": " is from the country of",
        "para_prefix": "The car brand ",
        "para_suffix": " comes from the country of",
    },
    "athlete_sport": {
        "prefix": "The athlete ",
        "suffix": " is famous for playing the sport of",
        "para_prefix": "Sports icon ",
        "para_suffix": " is a professional player of the sport of",
    },
    "historical_figure_country": {
        "prefix": "The historical figure ",
        "suffix": " was born in the country of",
        "para_prefix": "Famous figure ",
        "para_suffix": " came from the country of",
    },
}


ANSWER_ALIASES: Dict[str, List[str]] = {
    "United States": ["United States", "the United States", "USA", "U.S.", "US", "America"],
    "United Kingdom": ["United Kingdom", "the United Kingdom", "UK", "Britain", "Great Britain", "England"],
    "South Korea": ["South Korea", "Korea", "Republic of Korea"],
    "Netherlands": ["Netherlands", "the Netherlands", "Holland"],
    "Czech Republic": ["Czech Republic", "the Czech Republic", "Czechia"],
    "United Arab Emirates": ["United Arab Emirates", "the United Arab Emirates", "UAE"],
    "Philippines": ["Philippines", "the Philippines"],
    "Russia": ["Russia", "Russian Federation"],
    "China": ["China", "People's Republic of China"],
    "Australia": ["Australia", "Oceania"],
    "Slovenian": ["Slovenian", "Slovene"],
    "football": ["football", "soccer", "association football"],
    "American football": ["American football", "football"],
    "ice hockey": ["ice hockey", "hockey"],
    "track and field": ["track and field", "sprinting", "athletics", "running"],
    "pound": ["pound", "British pound", "pound sterling", "sterling", "Egyptian pound"],
    "dollar": ["dollar", "US dollar", "U.S. dollar", "United States dollar", "Australian dollar", "Canadian dollar", "New Zealand dollar", "Singapore dollar"],
    "euro": ["euro", "Euro"],
    "yen": ["yen", "Japanese yen"],
    "won": ["won", "South Korean won"],
    "rupee": ["rupee", "Indian rupee", "Pakistani rupee"],
    "peso": ["peso", "Mexican peso", "Argentine peso", "Chilean peso", "Colombian peso", "Philippine peso"],
    "franc": ["franc", "Swiss franc"],
    "krona": ["krona", "Swedish krona", "Icelandic krona"],
    "krone": ["krone", "Norwegian krone", "Danish krone"],
    "yuan": ["yuan", "renminbi", "Chinese yuan"],
    "lira": ["lira", "Turkish lira"],
    "ruble": ["ruble", "Russian ruble", "rouble"],
    "real": ["real", "Brazilian real"],
    "rand": ["rand", "South African rand"],
    "baht": ["baht", "Thai baht"],
    "ringgit": ["ringgit", "Malaysian ringgit"],
    "rupiah": ["rupiah", "Indonesian rupiah"],
    "shekel": ["shekel", "Israeli shekel", "Israeli new shekel", "new shekel"],
    "dirham": ["dirham", "UAE dirham", "Moroccan dirham"],
    "riyal": ["riyal", "Saudi riyal"],
    "dinar": ["dinar", "Kuwaiti dinar", "Jordanian dinar", "Iraqi dinar"],
    "zloty": ["zloty", "Polish zloty"],
    "forint": ["forint", "Hungarian forint"],
    "koruna": ["koruna", "Czech koruna"],
    "dong": ["dong", "Vietnamese dong"],
    "naira": ["naira", "Nigerian naira"],
    "shilling": ["shilling", "Kenyan shilling"],
    "taka": ["taka", "Bangladeshi taka"],
    "sol": ["sol", "Peruvian sol"],
    "leu": ["leu", "Romanian leu"],
    "lev": ["lev", "Bulgarian lev"],
    "hryvnia": ["hryvnia", "Ukrainian hryvnia"],
    "Mandarin": ["Mandarin", "Chinese", "Mandarin Chinese", "Standard Chinese"],
    "Spanish": ["Spanish", "Castilian"],
    "Portuguese": ["Portuguese"],
    "Arabic": ["Arabic"],
    "English": ["English"],
    "French": ["French"],
    "German": ["German"],
}


def get_raw_facts() -> Dict[str, List[Tuple[str, str]]]:
    """Return clean, unambiguous (subject, answer) pairs per relation."""
    return {
        "country_capital": [
            ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
            ("Portugal", "Lisbon"), ("United Kingdom", "London"), ("Ireland", "Dublin"),
            ("Netherlands", "Amsterdam"), ("Belgium", "Brussels"), ("Austria", "Vienna"),
            ("Poland", "Warsaw"), ("Czech Republic", "Prague"), ("Hungary", "Budapest"),
            ("Greece", "Athens"), ("Sweden", "Stockholm"), ("Norway", "Oslo"),
            ("Denmark", "Copenhagen"), ("Finland", "Helsinki"), ("Iceland", "Reykjavik"),
            ("Romania", "Bucharest"), ("Bulgaria", "Sofia"), ("Croatia", "Zagreb"),
            ("Serbia", "Belgrade"), ("Ukraine", "Kyiv"), ("Russia", "Moscow"),
            ("Turkey", "Ankara"), ("Egypt", "Cairo"), ("Morocco", "Rabat"),
            ("Algeria", "Algiers"), ("Tunisia", "Tunis"), ("Kenya", "Nairobi"),
            ("Ethiopia", "Addis Ababa"), ("Nigeria", "Abuja"), ("Ghana", "Accra"),
            ("Senegal", "Dakar"), ("Japan", "Tokyo"), ("South Korea", "Seoul"),
            ("China", "Beijing"), ("India", "New Delhi"), ("Thailand", "Bangkok"),
            ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("Malaysia", "Kuala Lumpur"),
            ("Philippines", "Manila"), ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"),
            ("Nepal", "Kathmandu"), ("Iran", "Tehran"), ("Iraq", "Baghdad"),
            ("Saudi Arabia", "Riyadh"), ("Jordan", "Amman"), ("Lebanon", "Beirut"),
            ("Oman", "Muscat"), ("Qatar", "Doha"), ("Kuwait", "Kuwait City"),
            ("Canada", "Ottawa"), ("Mexico", "Mexico City"), ("Cuba", "Havana"),
            ("Jamaica", "Kingston"), ("Panama", "Panama City"), ("Costa Rica", "San Jose"),
            ("Brazil", "Brasilia"), ("Argentina", "Buenos Aires"), ("Chile", "Santiago"),
            ("Peru", "Lima"), ("Colombia", "Bogota"), ("Ecuador", "Quito"),
            ("Uruguay", "Montevideo"), ("Paraguay", "Asuncion"), ("Venezuela", "Caracas"),
            ("Australia", "Canberra"), ("New Zealand", "Wellington"), ("Fiji", "Suva"),
            ("Slovakia", "Bratislava"), ("Slovenia", "Ljubljana"), ("Estonia", "Tallinn"),
            ("Latvia", "Riga"), ("Lithuania", "Vilnius"), ("Belarus", "Minsk"),
            ("Georgia", "Tbilisi"), ("Armenia", "Yerevan"), ("Azerbaijan", "Baku"),
            ("Kazakhstan", "Astana"), ("Uzbekistan", "Tashkent"), ("Mongolia", "Ulaanbaatar"),
            ("Cambodia", "Phnom Penh"), ("Laos", "Vientiane"), ("Afghanistan", "Kabul"),
            ("Sudan", "Khartoum"), ("Uganda", "Kampala"), ("Rwanda", "Kigali"),
            ("Zimbabwe", "Harare"), ("Zambia", "Lusaka"), ("Angola", "Luanda"),
        ],
        "country_continent": [
            ("France", "Europe"), ("Germany", "Europe"), ("Italy", "Europe"), ("Spain", "Europe"),
            ("Portugal", "Europe"), ("Poland", "Europe"), ("Sweden", "Europe"), ("Norway", "Europe"),
            ("Finland", "Europe"), ("Denmark", "Europe"), ("Ireland", "Europe"), ("Greece", "Europe"),
            ("Austria", "Europe"), ("Hungary", "Europe"), ("Romania", "Europe"), ("Belgium", "Europe"),
            ("Netherlands", "Europe"), ("Switzerland", "Europe"), ("Ukraine", "Europe"), ("Iceland", "Europe"),
            ("Croatia", "Europe"), ("Slovakia", "Europe"), ("Bulgaria", "Europe"), ("Serbia", "Europe"),
            ("Japan", "Asia"), ("China", "Asia"), ("India", "Asia"), ("South Korea", "Asia"),
            ("Thailand", "Asia"), ("Vietnam", "Asia"), ("Indonesia", "Asia"), ("Malaysia", "Asia"),
            ("Philippines", "Asia"), ("Pakistan", "Asia"), ("Bangladesh", "Asia"), ("Nepal", "Asia"),
            ("Iran", "Asia"), ("Iraq", "Asia"), ("Saudi Arabia", "Asia"), ("Jordan", "Asia"),
            ("Mongolia", "Asia"), ("Cambodia", "Asia"), ("Afghanistan", "Asia"), ("Oman", "Asia"),
            ("Kenya", "Africa"), ("Nigeria", "Africa"), ("Ethiopia", "Africa"), ("Morocco", "Africa"),
            ("Algeria", "Africa"), ("Tunisia", "Africa"), ("Ghana", "Africa"), ("Senegal", "Africa"),
            ("Uganda", "Africa"), ("Tanzania", "Africa"), ("South Africa", "Africa"), ("Zimbabwe", "Africa"),
            ("Zambia", "Africa"), ("Angola", "Africa"), ("Rwanda", "Africa"), ("Sudan", "Africa"),
            ("Brazil", "South America"), ("Argentina", "South America"), ("Chile", "South America"),
            ("Peru", "South America"), ("Colombia", "South America"), ("Ecuador", "South America"),
            ("Uruguay", "South America"), ("Paraguay", "South America"), ("Venezuela", "South America"),
            ("Bolivia", "South America"), ("Canada", "North America"), ("Mexico", "North America"),
            ("Cuba", "North America"), ("Jamaica", "North America"), ("Guatemala", "North America"),
            ("Honduras", "North America"), ("Costa Rica", "North America"), ("Panama", "North America"),
            ("Australia", "Australia"), ("New Zealand", "Oceania"), ("Fiji", "Oceania"),
            ("Papua New Guinea", "Oceania"),
        ],
        "country_currency": [
            ("Japan", "yen"), ("United Kingdom", "pound"), ("Switzerland", "franc"),
            ("Sweden", "krona"), ("Norway", "krone"), ("Denmark", "krone"),
            ("Iceland", "krona"), ("Poland", "zloty"), ("Hungary", "forint"),
            ("Czech Republic", "koruna"), ("Russia", "ruble"), ("Turkey", "lira"),
            ("India", "rupee"), ("Pakistan", "rupee"), ("China", "yuan"),
            ("South Korea", "won"), ("Thailand", "baht"), ("Malaysia", "ringgit"),
            ("Indonesia", "rupiah"), ("Vietnam", "dong"), ("Israel", "shekel"),
            ("Saudi Arabia", "riyal"), ("United Arab Emirates", "dirham"), ("Morocco", "dirham"),
            ("Kuwait", "dinar"), ("Jordan", "dinar"), ("Iraq", "dinar"),
            ("Brazil", "real"), ("South Africa", "rand"), ("Nigeria", "naira"),
            ("Kenya", "shilling"), ("Bangladesh", "taka"), ("Peru", "sol"),
            ("Mexico", "peso"), ("Argentina", "peso"), ("Chile", "peso"),
            ("Colombia", "peso"), ("Philippines", "peso"), ("United States", "dollar"),
            ("Canada", "dollar"), ("Australia", "dollar"), ("New Zealand", "dollar"),
            ("Singapore", "dollar"), ("France", "euro"), ("Germany", "euro"),
            ("Italy", "euro"), ("Spain", "euro"), ("Portugal", "euro"),
            ("Netherlands", "euro"), ("Belgium", "euro"), ("Austria", "euro"),
            ("Ireland", "euro"), ("Greece", "euro"), ("Finland", "euro"),
            ("Slovakia", "euro"), ("Slovenia", "euro"), ("Estonia", "euro"),
            ("Latvia", "euro"), ("Lithuania", "euro"), ("Croatia", "euro"),
            ("Egypt", "pound"), ("Romania", "leu"), ("Bulgaria", "lev"),
            ("Ukraine", "hryvnia"),
        ],
        "country_language": [
            ("France", "French"), ("Germany", "German"), ("Austria", "German"), ("Italy", "Italian"),
            ("Spain", "Spanish"), ("Mexico", "Spanish"), ("Argentina", "Spanish"), ("Colombia", "Spanish"),
            ("Peru", "Spanish"), ("Chile", "Spanish"), ("Ecuador", "Spanish"), ("Uruguay", "Spanish"),
            ("Venezuela", "Spanish"), ("Cuba", "Spanish"), ("Guatemala", "Spanish"), ("Costa Rica", "Spanish"),
            ("Panama", "Spanish"), ("Portugal", "Portuguese"), ("Brazil", "Portuguese"), ("Angola", "Portuguese"),
            ("Mozambique", "Portuguese"), ("United Kingdom", "English"), ("Australia", "English"),
            ("New Zealand", "English"), ("Jamaica", "English"), ("United States", "English"),
            ("Netherlands", "Dutch"), ("Poland", "Polish"), ("Czech Republic", "Czech"),
            ("Slovakia", "Slovak"), ("Hungary", "Hungarian"), ("Romania", "Romanian"),
            ("Bulgaria", "Bulgarian"), ("Greece", "Greek"), ("Sweden", "Swedish"),
            ("Norway", "Norwegian"), ("Denmark", "Danish"), ("Finland", "Finnish"),
            ("Iceland", "Icelandic"), ("Russia", "Russian"), ("Ukraine", "Ukrainian"),
            ("Serbia", "Serbian"), ("Croatia", "Croatian"), ("Slovenia", "Slovenian"),
            ("Estonia", "Estonian"), ("Latvia", "Latvian"), ("Lithuania", "Lithuanian"),
            ("Turkey", "Turkish"), ("Iran", "Persian"), ("Japan", "Japanese"),
            ("South Korea", "Korean"), ("China", "Mandarin"), ("Vietnam", "Vietnamese"),
            ("Thailand", "Thai"), ("Indonesia", "Indonesian"), ("Malaysia", "Malay"),
            ("Cambodia", "Khmer"), ("Bangladesh", "Bengali"), ("Nepal", "Nepali"),
            ("Egypt", "Arabic"), ("Saudi Arabia", "Arabic"), ("Iraq", "Arabic"),
            ("Jordan", "Arabic"), ("Morocco", "Arabic"), ("Algeria", "Arabic"),
            ("Tunisia", "Arabic"), ("Oman", "Arabic"), ("Qatar", "Arabic"),
            ("Kuwait", "Arabic"), ("Lebanon", "Arabic"), ("Israel", "Hebrew"),
            ("Georgia", "Georgian"), ("Armenia", "Armenian"), ("Mongolia", "Mongolian"),
        ],
        "us_state_capital": [
            ("Alabama", "Montgomery"), ("Alaska", "Juneau"), ("Arizona", "Phoenix"),
            ("Arkansas", "Little Rock"), ("California", "Sacramento"), ("Colorado", "Denver"),
            ("Connecticut", "Hartford"), ("Delaware", "Dover"), ("Florida", "Tallahassee"),
            ("Georgia", "Atlanta"), ("Hawaii", "Honolulu"), ("Idaho", "Boise"),
            ("Illinois", "Springfield"), ("Indiana", "Indianapolis"), ("Iowa", "Des Moines"),
            ("Kansas", "Topeka"), ("Kentucky", "Frankfort"), ("Louisiana", "Baton Rouge"),
            ("Maine", "Augusta"), ("Maryland", "Annapolis"), ("Massachusetts", "Boston"),
            ("Michigan", "Lansing"), ("Minnesota", "Saint Paul"), ("Mississippi", "Jackson"),
            ("Missouri", "Jefferson City"), ("Montana", "Helena"), ("Nebraska", "Lincoln"),
            ("Nevada", "Carson City"), ("New Hampshire", "Concord"), ("New Jersey", "Trenton"),
            ("New Mexico", "Santa Fe"), ("New York", "Albany"), ("North Carolina", "Raleigh"),
            ("North Dakota", "Bismarck"), ("Ohio", "Columbus"), ("Oklahoma", "Oklahoma City"),
            ("Oregon", "Salem"), ("Pennsylvania", "Harrisburg"), ("Rhode Island", "Providence"),
            ("South Carolina", "Columbia"), ("South Dakota", "Pierre"), ("Tennessee", "Nashville"),
            ("Texas", "Austin"), ("Utah", "Salt Lake City"), ("Vermont", "Montpelier"),
            ("Virginia", "Richmond"), ("Washington", "Olympia"), ("West Virginia", "Charleston"),
            ("Wisconsin", "Madison"), ("Wyoming", "Cheyenne"),
        ],
        "city_country": [
            ("Munich", "Germany"), ("Hamburg", "Germany"), ("Frankfurt", "Germany"), ("Cologne", "Germany"),
            ("Stuttgart", "Germany"), ("Lyon", "France"), ("Marseille", "France"), ("Toulouse", "France"),
            ("Nice", "France"), ("Bordeaux", "France"), ("Milan", "Italy"), ("Naples", "Italy"),
            ("Venice", "Italy"), ("Florence", "Italy"), ("Turin", "Italy"), ("Barcelona", "Spain"),
            ("Seville", "Spain"), ("Valencia", "Spain"), ("Bilbao", "Spain"), ("Porto", "Portugal"),
            ("Manchester", "United Kingdom"), ("Liverpool", "United Kingdom"), ("Edinburgh", "United Kingdom"),
            ("Glasgow", "United Kingdom"), ("Birmingham", "United Kingdom"), ("Rotterdam", "Netherlands"),
            ("Antwerp", "Belgium"), ("Zurich", "Switzerland"), ("Geneva", "Switzerland"), ("Basel", "Switzerland"),
            ("Salzburg", "Austria"), ("Krakow", "Poland"), ("Gdansk", "Poland"), ("Thessaloniki", "Greece"),
            ("Gothenburg", "Sweden"), ("Bergen", "Norway"), ("Aarhus", "Denmark"), ("Tampere", "Finland"),
            ("Istanbul", "Turkey"), ("Izmir", "Turkey"), ("Saint Petersburg", "Russia"), ("Novosibirsk", "Russia"),
            ("Osaka", "Japan"), ("Kyoto", "Japan"), ("Nagoya", "Japan"), ("Sapporo", "Japan"),
            ("Fukuoka", "Japan"), ("Yokohama", "Japan"), ("Busan", "South Korea"), ("Incheon", "South Korea"),
            ("Shanghai", "China"), ("Guangzhou", "China"), ("Shenzhen", "China"), ("Chengdu", "China"),
            ("Mumbai", "India"), ("Bengaluru", "India"), ("Kolkata", "India"), ("Chennai", "India"),
            ("Hyderabad", "India"), ("Karachi", "Pakistan"), ("Lahore", "Pakistan"), ("Chiang Mai", "Thailand"),
            ("Ho Chi Minh City", "Vietnam"), ("Surabaya", "Indonesia"), ("Cebu City", "Philippines"),
            ("Dubai", "United Arab Emirates"), ("Jeddah", "Saudi Arabia"), ("Alexandria", "Egypt"),
            ("Casablanca", "Morocco"), ("Marrakesh", "Morocco"), ("Lagos", "Nigeria"), ("Mombasa", "Kenya"),
            ("Johannesburg", "South Africa"), ("Cape Town", "South Africa"), ("Durban", "South Africa"),
            ("Toronto", "Canada"), ("Vancouver", "Canada"), ("Montreal", "Canada"), ("Calgary", "Canada"),
            ("Chicago", "United States"), ("Seattle", "United States"), ("Miami", "United States"),
            ("Houston", "United States"), ("Guadalajara", "Mexico"), ("Monterrey", "Mexico"),
            ("Rio de Janeiro", "Brazil"), ("Sao Paulo", "Brazil"), ("Medellin", "Colombia"),
            ("Cordoba", "Argentina"), ("Valparaiso", "Chile"), ("Guayaquil", "Ecuador"),
            ("Sydney", "Australia"), ("Melbourne", "Australia"), ("Brisbane", "Australia"),
            ("Perth", "Australia"), ("Auckland", "New Zealand"), ("Christchurch", "New Zealand"),
        ],
        "landmark_country": [
            ("Eiffel Tower", "France"), ("Louvre Museum", "France"), ("Palace of Versailles", "France"),
            ("Mont-Saint-Michel", "France"), ("Colosseum", "Italy"), ("Leaning Tower of Pisa", "Italy"),
            ("Trevi Fountain", "Italy"), ("Sagrada Familia", "Spain"), ("Alhambra", "Spain"),
            ("Brandenburg Gate", "Germany"), ("Neuschwanstein Castle", "Germany"), ("Cologne Cathedral", "Germany"),
            ("Big Ben", "United Kingdom"), ("Stonehenge", "United Kingdom"), ("Tower of London", "United Kingdom"),
            ("Buckingham Palace", "United Kingdom"), ("Acropolis of Athens", "Greece"), ("Parthenon", "Greece"),
            ("Hagia Sophia", "Turkey"), ("Blue Mosque", "Turkey"), ("Kremlin", "Russia"),
            ("Red Square", "Russia"), ("Great Wall", "China"), ("Forbidden City", "China"),
            ("Terracotta Army", "China"), ("Taj Mahal", "India"), ("Red Fort", "India"),
            ("Golden Temple", "India"), ("Mount Fuji", "Japan"), ("Fushimi Inari Shrine", "Japan"),
            ("Tokyo Tower", "Japan"), ("Gyeongbokgung Palace", "South Korea"), ("Angkor Wat", "Cambodia"),
            ("Borobudur", "Indonesia"), ("Petronas Towers", "Malaysia"), ("Burj Khalifa", "United Arab Emirates"),
            ("Petra", "Jordan"), ("Great Pyramid of Giza", "Egypt"), ("Great Sphinx of Giza", "Egypt"),
            ("Karnak Temple", "Egypt"), ("Table Mountain", "South Africa"), ("Statue of Liberty", "United States"),
            ("Golden Gate Bridge", "United States"), ("Mount Rushmore", "United States"),
            ("Empire State Building", "United States"), ("Grand Canyon", "United States"),
            ("CN Tower", "Canada"), ("Chichen Itza", "Mexico"), ("Teotihuacan", "Mexico"),
            ("Christ the Redeemer", "Brazil"), ("Sugarloaf Mountain", "Brazil"), ("Machu Picchu", "Peru"),
            ("Sydney Opera House", "Australia"), ("Uluru", "Australia"), ("Milford Sound", "New Zealand"),
            ("Atomium", "Belgium"), ("Schonbrunn Palace", "Austria"), ("Charles Bridge", "Czech Republic"),
            ("Sognefjord", "Norway"), ("Tivoli Gardens", "Denmark"), ("Blue Lagoon", "Iceland"),
        ],
        "element_symbol": [
            ("Hydrogen", "H"), ("Helium", "He"), ("Lithium", "Li"), ("Beryllium", "Be"),
            ("Boron", "B"), ("Carbon", "C"), ("Nitrogen", "N"), ("Oxygen", "O"),
            ("Fluorine", "F"), ("Neon", "Ne"), ("Sodium", "Na"), ("Magnesium", "Mg"),
            ("Aluminum", "Al"), ("Silicon", "Si"), ("Phosphorus", "P"), ("Sulfur", "S"),
            ("Chlorine", "Cl"), ("Argon", "Ar"), ("Potassium", "K"), ("Calcium", "Ca"),
            ("Scandium", "Sc"), ("Titanium", "Ti"), ("Vanadium", "V"), ("Chromium", "Cr"),
            ("Manganese", "Mn"), ("Iron", "Fe"), ("Cobalt", "Co"), ("Nickel", "Ni"),
            ("Copper", "Cu"), ("Zinc", "Zn"), ("Gallium", "Ga"), ("Germanium", "Ge"),
            ("Arsenic", "As"), ("Selenium", "Se"), ("Bromine", "Br"), ("Krypton", "Kr"),
            ("Rubidium", "Rb"), ("Strontium", "Sr"), ("Zirconium", "Zr"), ("Molybdenum", "Mo"),
            ("Palladium", "Pd"), ("Silver", "Ag"), ("Cadmium", "Cd"), ("Tin", "Sn"),
            ("Antimony", "Sb"), ("Tellurium", "Te"), ("Iodine", "I"), ("Xenon", "Xe"),
            ("Cesium", "Cs"), ("Barium", "Ba"), ("Tungsten", "W"), ("Osmium", "Os"),
            ("Iridium", "Ir"), ("Platinum", "Pt"), ("Gold", "Au"), ("Mercury", "Hg"),
            ("Lead", "Pb"), ("Bismuth", "Bi"), ("Polonium", "Po"), ("Radon", "Rn"),
            ("Radium", "Ra"), ("Thorium", "Th"), ("Uranium", "U"), ("Plutonium", "Pu"),
        ],
        "company_country": [
            ("Toyota", "Japan"), ("Sony", "Japan"), ("Nintendo", "Japan"), ("Panasonic", "Japan"),
            ("Canon", "Japan"), ("Fujitsu", "Japan"), ("Hitachi", "Japan"), ("SoftBank", "Japan"),
            ("Samsung", "South Korea"), ("LG Electronics", "South Korea"), ("SK Hynix", "South Korea"),
            ("Siemens", "Germany"), ("SAP", "Germany"), ("Bosch", "Germany"), ("Adidas", "Germany"),
            ("Allianz", "Germany"), ("Bayer", "Germany"), ("BASF", "Germany"),
            ("Nestle", "Switzerland"), ("Novartis", "Switzerland"), ("Roche", "Switzerland"), ("UBS", "Switzerland"),
            ("IKEA", "Sweden"), ("Ericsson", "Sweden"), ("Spotify", "Sweden"), ("H&M", "Sweden"),
            ("Nokia", "Finland"), ("Novo Nordisk", "Denmark"), ("Maersk", "Denmark"), ("Lego", "Denmark"),
            ("Equinor", "Norway"), ("Philips", "Netherlands"), ("ASML", "Netherlands"), ("Heineken", "Netherlands"),
            ("LVMH", "France"), ("TotalEnergies", "France"), ("Danone", "France"), ("Airbus", "France"),
            ("Sanofi", "France"), ("L'Oreal", "France"), ("Barclays", "United Kingdom"), ("Unilever", "United Kingdom"),
            ("AstraZeneca", "United Kingdom"), ("HSBC", "United Kingdom"), ("Vodafone", "United Kingdom"),
            ("Zara", "Spain"), ("Telefonica", "Spain"), ("Enel", "Italy"), ("Prada", "Italy"),
            ("Alibaba", "China"), ("Tencent", "China"), ("Huawei", "China"), ("Lenovo", "China"),
            ("Infosys", "India"), ("Tata Consultancy Services", "India"), ("TSMC", "Taiwan"), ("Asus", "Taiwan"),
            ("Shopify", "Canada"), ("BlackBerry", "Canada"), ("Apple", "United States"),
            ("Microsoft", "United States"), ("Amazon", "United States"), ("Intel", "United States"),
            ("Boeing", "United States"), ("Atlassian", "Australia"), ("BHP", "Australia"),
        ],
        "company_founder": [
            ("Microsoft", "Bill Gates"), ("Amazon", "Jeff Bezos"), ("Apple", "Steve Jobs"),
            ("Facebook", "Mark Zuckerberg"), ("Oracle", "Larry Ellison"), ("Dell", "Michael Dell"),
            ("Netflix", "Reed Hastings"), ("Nvidia", "Jensen Huang"), ("SpaceX", "Elon Musk"),
            ("Alibaba", "Jack Ma"), ("Tencent", "Pony Ma"), ("Huawei", "Ren Zhengfei"),
            ("Sony", "Akio Morita"), ("Honda", "Soichiro Honda"), ("Toyota", "Kiichiro Toyoda"),
            ("Panasonic", "Konosuke Matsushita"), ("IKEA", "Ingvar Kamprad"), ("Adidas", "Adi Dassler"),
            ("Puma", "Rudolf Dassler"), ("Porsche", "Ferdinand Porsche"), ("Ferrari", "Enzo Ferrari"),
            ("Ford Motor Company", "Henry Ford"), ("Walmart", "Sam Walton"), ("Disney", "Walt Disney"),
            ("Nike", "Phil Knight"), ("FedEx", "Fred Smith"), ("Bloomberg", "Michael Bloomberg"),
            ("Virgin Group", "Richard Branson"), ("Dyson", "James Dyson"), ("Chanel", "Coco Chanel"),
            ("Gucci", "Guccio Gucci"), ("Versace", "Gianni Versace"), ("Armani", "Giorgio Armani"),
            ("Salesforce", "Marc Benioff"), ("Zoom", "Eric Yuan"), ("Dropbox", "Drew Houston"),
            ("Airbnb", "Brian Chesky"), ("Uber", "Travis Kalanick"), ("Twitter", "Jack Dorsey"),
            ("LinkedIn", "Reid Hoffman"), ("Spotify", "Daniel Ek"), ("Shopify", "Tobias Lutke"),
            ("SoftBank", "Masayoshi Son"), ("Samsung", "Lee Byung-chul"), ("Hyundai", "Chung Ju-yung"),
            ("Nestle", "Henri Nestle"), ("Red Bull", "Dietrich Mateschitz"), ("LEGO", "Ole Kirk Christiansen"),
        ],
        "university_country": [
            ("Harvard University", "United States"), ("Stanford University", "United States"),
            ("Massachusetts Institute of Technology", "United States"), ("Yale University", "United States"),
            ("Princeton University", "United States"), ("Columbia University", "United States"),
            ("University of Oxford", "United Kingdom"), ("University of Cambridge", "United Kingdom"),
            ("Imperial College London", "United Kingdom"), ("University of Edinburgh", "United Kingdom"),
            ("University of Toronto", "Canada"), ("McGill University", "Canada"),
            ("University of British Columbia", "Canada"), ("University of Tokyo", "Japan"),
            ("Kyoto University", "Japan"), ("Osaka University", "Japan"),
            ("Seoul National University", "South Korea"), ("KAIST", "South Korea"),
            ("Tsinghua University", "China"), ("Peking University", "China"),
            ("Fudan University", "China"), ("National University of Singapore", "Singapore"),
            ("Nanyang Technological University", "Singapore"), ("University of Melbourne", "Australia"),
            ("University of Sydney", "Australia"), ("Australian National University", "Australia"),
            ("University of Auckland", "New Zealand"), ("ETH Zurich", "Switzerland"),
            ("EPFL", "Switzerland"), ("Sorbonne University", "France"),
            ("Ecole Polytechnique", "France"), ("Technical University of Munich", "Germany"),
            ("Heidelberg University", "Germany"), ("Humboldt University of Berlin", "Germany"),
            ("University of Bologna", "Italy"), ("Sapienza University of Rome", "Italy"),
            ("University of Amsterdam", "Netherlands"), ("Delft University of Technology", "Netherlands"),
            ("KU Leuven", "Belgium"), ("Uppsala University", "Sweden"),
            ("Karolinska Institute", "Sweden"), ("University of Copenhagen", "Denmark"),
            ("University of Oslo", "Norway"), ("University of Helsinki", "Finland"),
            ("University of Vienna", "Austria"), ("Charles University", "Czech Republic"),
            ("University of Warsaw", "Poland"), ("University of Barcelona", "Spain"),
            ("University of Coimbra", "Portugal"), ("Trinity College Dublin", "Ireland"),
            ("University of Sao Paulo", "Brazil"), ("University of Buenos Aires", "Argentina"),
            ("University of Cape Town", "South Africa"), ("Cairo University", "Egypt"),
            ("Hebrew University of Jerusalem", "Israel"), ("Moscow State University", "Russia"),
        ],
        "airline_country": [
            ("Lufthansa", "Germany"), ("Air France", "France"), ("British Airways", "United Kingdom"),
            ("Iberia", "Spain"), ("KLM", "Netherlands"), ("Swiss International Air Lines", "Switzerland"),
            ("Austrian Airlines", "Austria"), ("Finnair", "Finland"), ("Icelandair", "Iceland"),
            ("Aer Lingus", "Ireland"), ("TAP Air Portugal", "Portugal"), ("LOT Polish Airlines", "Poland"),
            ("Aegean Airlines", "Greece"), ("Turkish Airlines", "Turkey"), ("Aeroflot", "Russia"),
            ("Emirates", "United Arab Emirates"), ("Etihad Airways", "United Arab Emirates"),
            ("Qatar Airways", "Qatar"), ("Saudia", "Saudi Arabia"), ("El Al", "Israel"),
            ("Royal Jordanian", "Jordan"), ("Oman Air", "Oman"), ("EgyptAir", "Egypt"),
            ("Ethiopian Airlines", "Ethiopia"), ("Kenya Airways", "Kenya"), ("South African Airways", "South Africa"),
            ("Royal Air Maroc", "Morocco"), ("Japan Airlines", "Japan"), ("All Nippon Airways", "Japan"),
            ("Korean Air", "South Korea"), ("Asiana Airlines", "South Korea"), ("Air China", "China"),
            ("China Southern Airlines", "China"), ("Cathay Pacific", "Hong Kong"),
            ("Singapore Airlines", "Singapore"), ("Thai Airways", "Thailand"), ("Malaysia Airlines", "Malaysia"),
            ("Garuda Indonesia", "Indonesia"), ("Vietnam Airlines", "Vietnam"), ("Philippine Airlines", "Philippines"),
            ("Air India", "India"), ("Qantas", "Australia"), ("Air New Zealand", "New Zealand"),
            ("Air Canada", "Canada"), ("WestJet", "Canada"), ("Delta Air Lines", "United States"),
            ("United Airlines", "United States"), ("American Airlines", "United States"),
            ("Aeromexico", "Mexico"), ("Avianca", "Colombia"), ("Copa Airlines", "Panama"),
            ("Aerolineas Argentinas", "Argentina"),
        ],
        "car_brand_country": [
            ("Ferrari", "Italy"), ("Lamborghini", "Italy"), ("Maserati", "Italy"), ("Alfa Romeo", "Italy"),
            ("Fiat", "Italy"), ("Volkswagen", "Germany"), ("BMW", "Germany"), ("Mercedes-Benz", "Germany"),
            ("Audi", "Germany"), ("Porsche", "Germany"), ("Opel", "Germany"),
            ("Renault", "France"), ("Peugeot", "France"), ("Citroen", "France"), ("Bugatti", "France"),
            ("Volvo", "Sweden"), ("Saab", "Sweden"), ("Koenigsegg", "Sweden"),
            ("Rolls-Royce", "United Kingdom"), ("Bentley", "United Kingdom"), ("Aston Martin", "United Kingdom"),
            ("Jaguar", "United Kingdom"), ("Land Rover", "United Kingdom"), ("McLaren", "United Kingdom"),
            ("Skoda", "Czech Republic"), ("SEAT", "Spain"), ("Dacia", "Romania"),
            ("Honda", "Japan"), ("Nissan", "Japan"), ("Mazda", "Japan"), ("Subaru", "Japan"),
            ("Suzuki", "Japan"), ("Mitsubishi Motors", "Japan"), ("Lexus", "Japan"),
            ("Hyundai", "South Korea"), ("Kia", "South Korea"), ("Genesis", "South Korea"),
            ("Chevrolet", "United States"), ("Cadillac", "United States"), ("Jeep", "United States"),
            ("Tesla", "United States"), ("Chrysler", "United States"), ("Buick", "United States"),
            ("BYD", "China"), ("Geely", "China"), ("Tata Motors", "India"), ("Mahindra", "India"),
            ("Holden", "Australia"),
        ],
        "athlete_sport": [
            ("Michael Jordan", "basketball"), ("LeBron James", "basketball"), ("Kobe Bryant", "basketball"),
            ("Stephen Curry", "basketball"), ("Shaquille O'Neal", "basketball"), ("Magic Johnson", "basketball"),
            ("Larry Bird", "basketball"), ("Kareem Abdul-Jabbar", "basketball"),
            ("Lionel Messi", "football"), ("Cristiano Ronaldo", "football"), ("Pele", "football"),
            ("Diego Maradona", "football"), ("Zinedine Zidane", "football"), ("David Beckham", "football"),
            ("Kylian Mbappe", "football"), ("Neymar", "football"),
            ("Roger Federer", "tennis"), ("Rafael Nadal", "tennis"), ("Novak Djokovic", "tennis"),
            ("Serena Williams", "tennis"), ("Venus Williams", "tennis"), ("Steffi Graf", "tennis"),
            ("Pete Sampras", "tennis"), ("Andre Agassi", "tennis"),
            ("Tiger Woods", "golf"), ("Jack Nicklaus", "golf"), ("Arnold Palmer", "golf"),
            ("Rory McIlroy", "golf"), ("Phil Mickelson", "golf"),
            ("Tom Brady", "American football"), ("Peyton Manning", "American football"),
            ("Joe Montana", "American football"), ("Jerry Rice", "American football"),
            ("Babe Ruth", "baseball"), ("Jackie Robinson", "baseball"), ("Derek Jeter", "baseball"),
            ("Shohei Ohtani", "baseball"), ("Hank Aaron", "baseball"),
            ("Wayne Gretzky", "ice hockey"), ("Sidney Crosby", "ice hockey"), ("Mario Lemieux", "ice hockey"),
            ("Alexander Ovechkin", "ice hockey"),
            ("Muhammad Ali", "boxing"), ("Mike Tyson", "boxing"), ("Floyd Mayweather", "boxing"),
            ("Manny Pacquiao", "boxing"),
            ("Michael Phelps", "swimming"), ("Katie Ledecky", "swimming"), ("Ian Thorpe", "swimming"),
            ("Usain Bolt", "track and field"), ("Carl Lewis", "track and field"), ("Jesse Owens", "track and field"),
            ("Simone Biles", "gymnastics"), ("Nadia Comaneci", "gymnastics"),
            ("Sachin Tendulkar", "cricket"), ("Virat Kohli", "cricket"), ("Don Bradman", "cricket"),
            ("Lewis Hamilton", "Formula One"), ("Michael Schumacher", "Formula One"), ("Ayrton Senna", "Formula One"),
            ("Max Verstappen", "Formula One"),
        ],
        "historical_figure_country": [
            ("Albert Einstein", "Germany"), ("Johann Sebastian Bach", "Germany"), ("Ludwig van Beethoven", "Germany"),
            ("Johannes Gutenberg", "Germany"), ("Karl Marx", "Germany"), ("Martin Luther", "Germany"),
            ("Isaac Newton", "United Kingdom"), ("Charles Darwin", "United Kingdom"), ("Alan Turing", "United Kingdom"),
            ("William Shakespeare", "United Kingdom"), ("Michael Faraday", "United Kingdom"), ("Florence Nightingale", "United Kingdom"),
            ("Marie Curie", "Poland"), ("Frederic Chopin", "Poland"), ("Nicolaus Copernicus", "Poland"),
            ("Wolfgang Amadeus Mozart", "Austria"), ("Sigmund Freud", "Austria"), ("Franz Schubert", "Austria"),
            ("Leonardo da Vinci", "Italy"), ("Galileo Galilei", "Italy"), ("Michelangelo", "Italy"),
            ("Marco Polo", "Italy"), ("Antonio Vivaldi", "Italy"), ("Giuseppe Verdi", "Italy"),
            ("Napoleon Bonaparte", "France"), ("Louis Pasteur", "France"), ("Victor Hugo", "France"),
            ("Rene Descartes", "France"), ("Claude Monet", "France"), ("Joan of Arc", "France"),
            ("Pablo Picasso", "Spain"), ("Salvador Dali", "Spain"), ("Miguel de Cervantes", "Spain"),
            ("Vincent van Gogh", "Netherlands"), ("Rembrandt", "Netherlands"), ("Anne Frank", "Germany"),
            ("Alfred Nobel", "Sweden"), ("Carl Linnaeus", "Sweden"), ("Hans Christian Andersen", "Denmark"),
            ("Niels Bohr", "Denmark"), ("Edvard Munch", "Norway"), ("Henrik Ibsen", "Norway"),
            ("Leo Tolstoy", "Russia"), ("Fyodor Dostoevsky", "Russia"), ("Pyotr Ilyich Tchaikovsky", "Russia"),
            ("Dmitri Mendeleev", "Russia"), ("Mahatma Gandhi", "India"), ("Rabindranath Tagore", "India"),
            ("Confucius", "China"), ("Sun Tzu", "China"), ("Akira Kurosawa", "Japan"),
            ("Nelson Mandela", "South Africa"), ("Cleopatra", "Egypt"), ("Aristotle", "Greece"),
            ("Socrates", "Greece"), ("Archimedes", "Italy"), ("George Washington", "United States"),
            ("Abraham Lincoln", "United States"), ("Thomas Edison", "United States"), ("Benjamin Franklin", "United States"),
            ("Nikola Tesla", "Croatia"), ("Frida Kahlo", "Mexico"), ("Simon Bolivar", "Venezuela"),
            ("Vasco da Gama", "Portugal"), ("Ferdinand Magellan", "Portugal"),
        ],
    }


def get_2hop_facts() -> List[Dict[str, str]]:
    city_to_country = [
        ("Munich", "Germany"), ("Hamburg", "Germany"), ("Frankfurt", "Germany"),
        ("Lyon", "France"), ("Marseille", "France"), ("Nice", "France"),
        ("Milan", "Italy"), ("Venice", "Italy"), ("Florence", "Italy"), ("Naples", "Italy"),
        ("Barcelona", "Spain"), ("Seville", "Spain"), ("Valencia", "Spain"),
        ("Porto", "Portugal"), ("Manchester", "United Kingdom"), ("Edinburgh", "United Kingdom"),
        ("Liverpool", "United Kingdom"), ("Rotterdam", "Netherlands"), ("Antwerp", "Belgium"),
        ("Salzburg", "Austria"), ("Krakow", "Poland"), ("Thessaloniki", "Greece"),
        ("Gothenburg", "Sweden"), ("Bergen", "Norway"), ("Istanbul", "Turkey"),
        ("Saint Petersburg", "Russia"), ("Osaka", "Japan"), ("Kyoto", "Japan"),
        ("Nagoya", "Japan"), ("Busan", "South Korea"), ("Shanghai", "China"),
        ("Guangzhou", "China"), ("Mumbai", "India"), ("Bengaluru", "India"),
        ("Karachi", "Pakistan"), ("Chiang Mai", "Thailand"), ("Ho Chi Minh City", "Vietnam"),
        ("Surabaya", "Indonesia"), ("Alexandria", "Egypt"), ("Casablanca", "Morocco"),
        ("Lagos", "Nigeria"), ("Mombasa", "Kenya"), ("Toronto", "Canada"),
        ("Vancouver", "Canada"), ("Montreal", "Canada"), ("Guadalajara", "Mexico"),
        ("Rio de Janeiro", "Brazil"), ("Sao Paulo", "Brazil"), ("Medellin", "Colombia"),
        ("Sydney", "Australia"), ("Melbourne", "Australia"), ("Auckland", "New Zealand"),
    ]
    landmark_to_country = [
        ("Eiffel Tower", "France"), ("Louvre Museum", "France"), ("Colosseum", "Italy"),
        ("Leaning Tower of Pisa", "Italy"), ("Sagrada Familia", "Spain"), ("Alhambra", "Spain"),
        ("Brandenburg Gate", "Germany"), ("Neuschwanstein Castle", "Germany"),
        ("Big Ben", "United Kingdom"), ("Stonehenge", "United Kingdom"),
        ("Acropolis of Athens", "Greece"), ("Hagia Sophia", "Turkey"),
        ("Great Wall", "China"), ("Forbidden City", "China"), ("Taj Mahal", "India"),
        ("Mount Fuji", "Japan"), ("Angkor Wat", "Cambodia"), ("Borobudur", "Indonesia"),
        ("Petronas Towers", "Malaysia"), ("Petra", "Jordan"),
        ("Great Pyramid of Giza", "Egypt"), ("CN Tower", "Canada"),
        ("Chichen Itza", "Mexico"), ("Christ the Redeemer", "Brazil"),
        ("Machu Picchu", "Peru"), ("Sydney Opera House", "Australia"),
        ("Uluru", "Australia"), ("Charles Bridge", "Czech Republic"),
    ]
    company_to_country = [
        ("Toyota", "Japan"), ("Sony", "Japan"), ("Nintendo", "Japan"),
        ("Samsung", "South Korea"), ("Siemens", "Germany"), ("Adidas", "Germany"),
        ("IKEA", "Sweden"), ("Spotify", "Sweden"), ("Nokia", "Finland"),
        ("Lego", "Denmark"), ("Philips", "Netherlands"), ("LVMH", "France"),
        ("Unilever", "United Kingdom"), ("Alibaba", "China"), ("Infosys", "India"),
        ("Shopify", "Canada"), ("Atlassian", "Australia"),
    ]
    country_to_capital = {
        "Germany": "Berlin", "France": "Paris", "Italy": "Rome", "Spain": "Madrid",
        "Portugal": "Lisbon", "United Kingdom": "London", "Netherlands": "Amsterdam",
        "Belgium": "Brussels", "Austria": "Vienna", "Poland": "Warsaw", "Greece": "Athens",
        "Sweden": "Stockholm", "Norway": "Oslo", "Turkey": "Ankara", "Russia": "Moscow",
        "Japan": "Tokyo", "South Korea": "Seoul", "China": "Beijing", "India": "New Delhi",
        "Pakistan": "Islamabad", "Thailand": "Bangkok", "Vietnam": "Hanoi",
        "Indonesia": "Jakarta", "Egypt": "Cairo", "Morocco": "Rabat", "Nigeria": "Abuja",
        "Kenya": "Nairobi", "Canada": "Ottawa", "Mexico": "Mexico City", "Brazil": "Brasilia",
        "Colombia": "Bogota", "Australia": "Canberra", "New Zealand": "Wellington",
        "Cambodia": "Phnom Penh", "Malaysia": "Kuala Lumpur", "Jordan": "Amman",
        "Peru": "Lima", "Czech Republic": "Prague", "Finland": "Helsinki", "Denmark": "Copenhagen",
    }
    country_to_currency = {
        "Japan": "yen", "United Kingdom": "pound", "Sweden": "krona", "Norway": "krone",
        "Poland": "zloty", "Russia": "ruble", "Turkey": "lira", "India": "rupee",
        "China": "yuan", "South Korea": "won", "Thailand": "baht", "Vietnam": "dong",
        "Indonesia": "rupiah", "Brazil": "real", "Mexico": "peso", "Canada": "dollar",
        "Australia": "dollar", "Denmark": "krone", "Czech Republic": "koruna",
        "Malaysia": "ringgit", "Jordan": "dinar", "Peru": "sol",
    }
    country_to_language = {
        "Germany": "German", "France": "French", "Italy": "Italian", "Spain": "Spanish",
        "Portugal": "Portuguese", "Netherlands": "Dutch", "Poland": "Polish",
        "Greece": "Greek", "Sweden": "Swedish", "Norway": "Norwegian", "Turkey": "Turkish",
        "Russia": "Russian", "Japan": "Japanese", "South Korea": "Korean",
        "China": "Mandarin", "Thailand": "Thai", "Vietnam": "Vietnamese",
        "Indonesia": "Indonesian", "Egypt": "Arabic", "Brazil": "Portuguese",
        "Mexico": "Spanish", "Finland": "Finnish", "Denmark": "Danish",
    }

    items: List[Dict[str, str]] = []
    idx = 0

    for city, country in city_to_country:
        cap = country_to_capital[country]
        prefix = "The city of "
        suffix = " is in a country whose capital is"
        items.append({
            "id": f"2hop_{idx:04d}",
            "hop_type": "city_country_capital",
            "subject": city,
            "bridge_entity": country,
            "bridge_relation": "city_country",
            "hop2_relation": "country_capital",
            "answer": cap,
            "aliases": list(dict.fromkeys([cap] + ANSWER_ALIASES.get(cap, []))),
            "prefix": prefix,
            "suffix": suffix,
            "direct_prompt": prefix + city + suffix,
            "hop1_prompt": f"The city of {city} is in the country of",
            "hop2_prompt": f"The capital of {country} is",
            "hop2_prefix": "The capital of ",
            "hop2_suffix": " is",
        })
        idx += 1

    for lm, country in landmark_to_country:
        cap = country_to_capital[country]
        prefix = "The landmark "
        suffix = " is in a country whose capital is"
        items.append({
            "id": f"2hop_{idx:04d}",
            "hop_type": "landmark_country_capital",
            "subject": lm,
            "bridge_entity": country,
            "bridge_relation": "landmark_country",
            "hop2_relation": "country_capital",
            "answer": cap,
            "aliases": list(dict.fromkeys([cap] + ANSWER_ALIASES.get(cap, []))),
            "prefix": prefix,
            "suffix": suffix,
            "direct_prompt": prefix + lm + suffix,
            "hop1_prompt": f"The landmark {lm} is in the country of",
            "hop2_prompt": f"The capital of {country} is",
            "hop2_prefix": "The capital of ",
            "hop2_suffix": " is",
        })
        idx += 1

    for lm, country in landmark_to_country:
        if country in country_to_currency:
            curr = country_to_currency[country]
            prefix = "The landmark "
            suffix = " is in a country whose official currency is the"
            items.append({
                "id": f"2hop_{idx:04d}",
                "hop_type": "landmark_country_currency",
                "subject": lm,
                "bridge_entity": country,
                "bridge_relation": "landmark_country",
                "hop2_relation": "country_currency",
                "answer": curr,
                "aliases": list(dict.fromkeys([curr] + ANSWER_ALIASES.get(curr, []))),
                "prefix": prefix,
                "suffix": suffix,
                "direct_prompt": prefix + lm + suffix,
                "hop1_prompt": f"The landmark {lm} is in the country of",
                "hop2_prompt": f"The official currency of {country} is the",
                "hop2_prefix": "The official currency of ",
                "hop2_suffix": " is the",
            })
            idx += 1

    for comp, country in company_to_country:
        cap = country_to_capital[country]
        prefix = "The company "
        suffix = " is based in a country whose capital is"
        items.append({
            "id": f"2hop_{idx:04d}",
            "hop_type": "company_country_capital",
            "subject": comp,
            "bridge_entity": country,
            "bridge_relation": "company_country",
            "hop2_relation": "country_capital",
            "answer": cap,
            "aliases": list(dict.fromkeys([cap] + ANSWER_ALIASES.get(cap, []))),
            "prefix": prefix,
            "suffix": suffix,
            "direct_prompt": prefix + comp + suffix,
            "hop1_prompt": f"The company {comp} is headquartered in the country of",
            "hop2_prompt": f"The capital of {country} is",
            "hop2_prefix": "The capital of ",
            "hop2_suffix": " is",
        })
        idx += 1

    for city, country in city_to_country[:30]:
        if country in country_to_language:
            lang = country_to_language[country]
            prefix = "The city of "
            suffix = " is in a country whose main language is"
            items.append({
                "id": f"2hop_{idx:04d}",
                "hop_type": "city_country_language",
                "subject": city,
                "bridge_entity": country,
                "bridge_relation": "city_country",
                "hop2_relation": "country_language",
                "answer": lang,
                "aliases": list(dict.fromkeys([lang] + ANSWER_ALIASES.get(lang, []))),
                "prefix": prefix,
                "suffix": suffix,
                "direct_prompt": prefix + city + suffix,
                "hop1_prompt": f"The city of {city} is in the country of",
                "hop2_prompt": f"The primary official language of {country} is",
                "hop2_prefix": "The primary official language of ",
                "hop2_suffix": " is",
            })
            idx += 1

    rng = random.Random(SEED + 2)
    bridges = sorted({x["bridge_entity"] for x in items})
    rng.shuffle(bridges)
    n_b = len(bridges)
    n_tr = int(round(0.6 * n_b))
    n_va = int(round(0.2 * n_b))
    bridge_split: Dict[str, str] = {}
    for i, b in enumerate(bridges):
        if i < n_tr:
            bridge_split[b] = "train"
        elif i < n_tr + n_va:
            bridge_split[b] = "val"
        else:
            bridge_split[b] = "test"
    for item in items:
        item["split"] = bridge_split[item["bridge_entity"]]
        item["direct_subj_char_span"] = [len(item["prefix"]), len(item["prefix"]) + len(item["subject"])]

    return items


def build_datasets(out_dir: Path) -> Tuple[dict, dict]:
    raw_by_rel = get_raw_facts()
    rng = random.Random(SEED)

    cleaned_by_rel: Dict[str, List[Tuple[str, str]]] = {}
    subj_to_rels: Dict[str, List[str]] = defaultdict(list)

    for rel, pairs in raw_by_rel.items():
        seen_s = set()
        clean_pairs = []
        for s, a in pairs:
            ns = _norm(s)
            if ns in seen_s:
                continue
            seen_s.add(ns)
            clean_pairs.append((s, a))
            subj_to_rels[ns].append(rel)
        cleaned_by_rel[rel] = clean_pairs

    primary_rel_to_subjs: Dict[str, List[str]] = defaultdict(list)
    for ns, rels in sorted(subj_to_rels.items()):
        primary_rel_to_subjs[rels[0]].append(ns)

    subj_split: Dict[str, str] = {}
    for rel in sorted(primary_rel_to_subjs.keys()):
        subjs = list(primary_rel_to_subjs[rel])
        rng.shuffle(subjs)
        n = len(subjs)
        n_train = int(round(0.60 * n))
        n_val = int(round(0.20 * n))
        for i, ns in enumerate(subjs):
            if i < n_train:
                subj_split[ns] = "train"
            elif i < n_train + n_val:
                subj_split[ns] = "val"
            else:
                subj_split[ns] = "test"

    facts: List[dict] = []
    relations_sorted = sorted(cleaned_by_rel.keys())
    rel_to_fold = {rel: idx for idx, rel in enumerate(relations_sorted)}

    rel_candidate_answers: Dict[str, List[str]] = {}
    for rel in relations_sorted:
        uniq_ans = []
        seen_a = set()
        for _, a in cleaned_by_rel[rel]:
            na = _norm(a)
            if na not in seen_a:
                seen_a.add(na)
                uniq_ans.append(a)
        rel_candidate_answers[rel] = uniq_ans[:50]

    fact_idx = 0
    for rel in relations_sorted:
        tmpl = RELATION_TEMPLATES[rel]
        pairs = cleaned_by_rel[rel]
        train_demos = [(s, a) for (s, a) in pairs if subj_split[_norm(s)] == "train"]

        for s, a in pairs:
            ns = _norm(s)
            na = _norm(a)
            split = subj_split[ns]

            demo_pool = [(ds, da) for (ds, da) in train_demos if _norm(ds) != ns and _norm(da) != na]
            h = int(hashlib.sha256(f"{rel}::{s}".encode("utf-8")).hexdigest()[:8], 16)
            drng = random.Random(h)
            drng.shuffle(demo_pool)
            chosen_demos = []
            used_ans = set()
            for ds, da in demo_pool:
                if _norm(da) not in used_ans:
                    chosen_demos.append((ds, da))
                    used_ans.add(_norm(da))
                if len(chosen_demos) == 3:
                    break
            if len(chosen_demos) < 3:
                for ds, da in demo_pool:
                    if (ds, da) not in chosen_demos:
                        chosen_demos.append((ds, da))
                    if len(chosen_demos) == 3:
                        break

            icl_lines = [
                f"{tmpl['prefix']}{ds}{tmpl['suffix']} {da}."
                for (ds, da) in chosen_demos
            ]
            icl3_prefix = "\n".join(icl_lines) + "\n" + tmpl["prefix"]

            direct_prompt = tmpl["prefix"] + s + tmpl["suffix"]
            paraphrase_prompt = tmpl["para_prefix"] + s + tmpl["para_suffix"]
            icl3_prompt = icl3_prefix + s + tmpl["suffix"]

            aliases = list(dict.fromkeys([a] + ANSWER_ALIASES.get(a, [])))

            facts.append({
                "idx": fact_idx,
                "id": f"{rel}_{fact_idx:04d}",
                "relation": rel,
                "loro_fold": rel_to_fold[rel],
                "subject": s,
                "subject_norm": ns,
                "answer": a,
                "aliases": aliases,
                "split": split,
                "prefix": tmpl["prefix"],
                "suffix": tmpl["suffix"],
                "direct_prompt": direct_prompt,
                "direct_subj_char_span": [len(tmpl["prefix"]), len(tmpl["prefix"]) + len(s)],
                "para_prefix": tmpl["para_prefix"],
                "para_suffix": tmpl["para_suffix"],
                "paraphrase_prompt": paraphrase_prompt,
                "para_subj_char_span": [len(tmpl["para_prefix"]), len(tmpl["para_prefix"]) + len(s)],
                "icl3_prefix": icl3_prefix,
                "icl3_prompt": icl3_prompt,
                "icl3_subj_char_span": [len(icl3_prefix), len(icl3_prefix) + len(s)],
                "icl3_demo_subjects": [ds for ds, _ in chosen_demos],
            })
            fact_idx += 1

    split_subjs = defaultdict(set)
    for f in facts:
        split_subjs[f["split"]].add(f["subject_norm"])
    assert not (split_subjs["train"] & split_subjs["val"]), "Subject overlap train/val!"
    assert not (split_subjs["train"] & split_subjs["test"]), "Subject overlap train/test!"
    assert not (split_subjs["val"] & split_subjs["test"]), "Subject overlap val/test!"

    for f in facts:
        for ds in f["icl3_demo_subjects"]:
            nds = _norm(ds)
            assert subj_split[nds] == "train", f"ICL demo {ds} not from train split!"
            assert nds != f["subject_norm"], f"ICL demo matches query subject {ds}!"
        p = f["direct_prompt"]
        a0, a1 = f["direct_subj_char_span"]
        assert p[a0:a1] == f["subject"]
        pp = f["paraphrase_prompt"]
        b0, b1 = f["para_subj_char_span"]
        assert pp[b0:b1] == f["subject"]
        ip = f["icl3_prompt"]
        c0, c1 = f["icl3_subj_char_span"]
        assert ip[c0:c1] == f["subject"]

    split_counts = defaultdict(int)
    rel_counts = defaultdict(lambda: defaultdict(int))
    for f in facts:
        split_counts[f["split"]] += 1
        rel_counts[f["relation"]][f["split"]] += 1

    loro_folds = []
    for rel in relations_sorted:
        fold_id = rel_to_fold[rel]
        loro_folds.append({
            "fold_id": fold_id,
            "held_out_relation": rel,
            "train_relations": [r for r in relations_sorted if r != rel],
            "n_held_out_facts": sum(rel_counts[rel].values()),
            "n_held_out_test_split": rel_counts[rel]["test"],
        })

    dataset_obj = {
        "meta": {
            "name": "platos_key_factual_benchmark_v1",
            "seed": SEED,
            "n_facts": len(facts),
            "n_relations": len(relations_sorted),
            "relations": relations_sorted,
            "split_counts": dict(split_counts),
            "relation_split_counts": {r: dict(rel_counts[r]) for r in relations_sorted},
            "subject_disjoint_globally": True,
            "icl3_demos_strictly_from_train": True,
        },
        "relation_candidate_answers": rel_candidate_answers,
        "loro_folds": loro_folds,
        "facts": facts,
    }

    facts_2hop = get_2hop_facts()
    hop2_split_counts = defaultdict(int)
    hop2_type_counts = defaultdict(int)
    for x in facts_2hop:
        hop2_split_counts[x["split"]] += 1
        hop2_type_counts[x["hop_type"]] += 1
        assert x["direct_prompt"] == x["prefix"] + x["subject"] + x["suffix"]

    dataset_2hop_obj = {
        "meta": {
            "name": "platos_key_2hop_bridge_v1",
            "seed": SEED + 2,
            "n_items": len(facts_2hop),
            "hop_type_counts": dict(hop2_type_counts),
            "split_counts": dict(hop2_split_counts),
            "bridge_disjoint_splits": True,
        },
        "items": facts_2hop,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_obj, f, indent=2, ensure_ascii=False)
    with open(out_dir / "dataset_2hop.json", "w", encoding="utf-8") as f:
        json.dump(dataset_2hop_obj, f, indent=2, ensure_ascii=False)

    return dataset_obj, dataset_2hop_obj


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    ds, ds2 = build_datasets(here)
    print(f"Wrote dataset.json: {ds['meta']['n_facts']} facts across {ds['meta']['n_relations']} relations.")
    print("Split counts:", ds["meta"]["split_counts"])
    for rel, sc in ds["meta"]["relation_split_counts"].items():
        tot = sum(sc.values())
        print(f"  {rel:28s}: total={tot:3d} (train={sc.get('train',0):2d}, val={sc.get('val',0):2d}, test={sc.get('test',0):2d})")
    print(f"Wrote dataset_2hop.json: {ds2['meta']['n_items']} 2-hop items. Splits:", ds2["meta"]["split_counts"])
