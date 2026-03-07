"""
add_language_families.py

Adds/updates the 'family' column in linglist.csv using:
  1. A hardcoded lookup for common languages (fast, offline)
  2. Wikidata SPARQL for anything still unclassified (free, comprehensive)

The SPARQL query retrieves the TOP-LEVEL language family for each ISO 639-2
code by traversing the P279 (subclass-of) chain up to the root family
(a Q25295 entity that has no parent that is also Q25295).

Run once (or again after updating linglist.csv):
    python pipeline/add_language_families.py
"""

import os
import re
import time
import requests
import pandas as pd

_DIR = os.path.dirname(__file__)

# ── 1. Hardcoded lookup (ISO 639-2 → family) ─────────────────────────────────
FAMILY_BY_CODE: dict[str, str] = {
    # Indo-European — Germanic
    "eng": "Indo-European", "deu": "Indo-European", "nld": "Indo-European",
    "swe": "Indo-European", "nor": "Indo-European", "dan": "Indo-European",
    "isl": "Indo-European", "fao": "Indo-European", "yid": "Indo-European",
    "afr": "Indo-European", "ltz": "Indo-European", "got": "Indo-European",
    "ang": "Indo-European", "enm": "Indo-European", "nds": "Indo-European",
    # Indo-European — Romance
    "fra": "Indo-European", "spa": "Indo-European", "por": "Indo-European",
    "ita": "Indo-European", "ron": "Indo-European", "cat": "Indo-European",
    "glg": "Indo-European", "oci": "Indo-European", "lat": "Indo-European",
    "fro": "Indo-European", "pro": "Indo-European", "roa": "Indo-European",
    "mol": "Indo-European", "lad": "Indo-European", "frm": "Indo-European",
    "arg": "Indo-European", "ast": "Indo-European", "scn": "Indo-European",
    "vec": "Indo-European", "nap": "Indo-European", "lmo": "Indo-European",
    "fur": "Indo-European", "lij": "Indo-European", "pms": "Indo-European",
    # Indo-European — Slavic
    "rus": "Indo-European", "pol": "Indo-European", "ces": "Indo-European",
    "slk": "Indo-European", "bul": "Indo-European", "hrv": "Indo-European",
    "srp": "Indo-European", "bos": "Indo-European", "slv": "Indo-European",
    "mkd": "Indo-European", "ukr": "Indo-European", "bel": "Indo-European",
    "sla": "Indo-European", "chu": "Indo-European", "hsb": "Indo-European",
    "dsb": "Indo-European", "csb": "Indo-European", "rue": "Indo-European",
    # Indo-European — Indo-Iranian
    "hin": "Indo-European", "urd": "Indo-European", "ben": "Indo-European",
    "pan": "Indo-European", "guj": "Indo-European", "mar": "Indo-European",
    "nep": "Indo-European", "sin": "Indo-European", "fas": "Indo-European",
    "per": "Indo-European", "pus": "Indo-European", "bal": "Indo-European",
    "oss": "Indo-European", "san": "Indo-European", "pli": "Indo-European",
    "pra": "Indo-European", "rom": "Indo-European", "mai": "Indo-European",
    "bho": "Indo-European", "awa": "Indo-European", "raj": "Indo-European",
    "doi": "Indo-European", "kok": "Indo-European", "anp": "Indo-European",
    "bih": "Indo-European", "asm": "Indo-European", "ori": "Indo-European",
    "ave": "Indo-European", "peo": "Indo-European", "xpr": "Indo-European",
    "kho": "Indo-European", "wbl": "Indo-European",
    # Indo-European — Baltic
    "lit": "Indo-European", "lav": "Indo-European", "bat": "Indo-European",
    "prg": "Indo-European",
    # Indo-European — Celtic
    "gle": "Indo-European", "wel": "Indo-European", "bre": "Indo-European",
    "glv": "Indo-European", "gla": "Indo-European", "cor": "Indo-European",
    "cel": "Indo-European", "wls": "Indo-European",
    # Indo-European — Greek
    "ell": "Indo-European", "grc": "Indo-European", "grk": "Indo-European",
    # Indo-European — Armenian, Albanian
    "hye": "Indo-European", "sqi": "Indo-European",
    "arm": "Indo-European", "alb": "Indo-European",
    # Indo-European — other / old codes
    "toc": "Indo-European", "hit": "Indo-European", "ine": "Indo-European",
    "gem": "Indo-European", "ira": "Indo-European", "inc": "Indo-European",
    "ice": "Indo-European", "rum": "Indo-European", "scc": "Indo-European",
    "scr": "Indo-European", "mac": "Indo-European",

    # Sino-Tibetan
    "zho": "Sino-Tibetan", "chi": "Sino-Tibetan", "cmn": "Sino-Tibetan",
    "yue": "Sino-Tibetan", "wuu": "Sino-Tibetan", "nan": "Sino-Tibetan",
    "hak": "Sino-Tibetan", "bod": "Sino-Tibetan", "tib": "Sino-Tibetan",
    "mya": "Sino-Tibetan", "dzo": "Sino-Tibetan", "sit": "Sino-Tibetan",
    "lus": "Sino-Tibetan", "bai": "Sino-Tibetan", "kac": "Sino-Tibetan",
    "lhu": "Sino-Tibetan", "hni": "Sino-Tibetan",

    # Afro-Asiatic
    "ara": "Afro-Asiatic", "heb": "Afro-Asiatic", "mlt": "Afro-Asiatic",
    "amh": "Afro-Asiatic", "orm": "Afro-Asiatic", "som": "Afro-Asiatic",
    "hau": "Afro-Asiatic", "tir": "Afro-Asiatic", "tig": "Afro-Asiatic",
    "gez": "Afro-Asiatic", "aar": "Afro-Asiatic", "bej": "Afro-Asiatic",
    "sid": "Afro-Asiatic", "afa": "Afro-Asiatic", "sem": "Afro-Asiatic",
    "cus": "Afro-Asiatic", "cop": "Afro-Asiatic", "egy": "Afro-Asiatic",
    "syc": "Afro-Asiatic", "syr": "Afro-Asiatic", "arc": "Afro-Asiatic",
    "akk": "Afro-Asiatic", "phn": "Afro-Asiatic", "nqo": "Afro-Asiatic",
    "ber": "Afro-Asiatic", "kab": "Afro-Asiatic", "shy": "Afro-Asiatic",
    "byn": "Afro-Asiatic", "afb": "Afro-Asiatic", "apc": "Afro-Asiatic",
    "arb": "Afro-Asiatic", "ary": "Afro-Asiatic", "aeb": "Afro-Asiatic",
    "acm": "Afro-Asiatic", "shu": "Afro-Asiatic", "lkt": "Afro-Asiatic",

    # Niger-Congo
    "swa": "Niger-Congo", "yor": "Niger-Congo", "ibo": "Niger-Congo",
    "aka": "Niger-Congo", "zul": "Niger-Congo", "xho": "Niger-Congo",
    "sot": "Niger-Congo", "tsn": "Niger-Congo", "sna": "Niger-Congo",
    "nya": "Niger-Congo", "kin": "Niger-Congo", "run": "Niger-Congo",
    "lin": "Niger-Congo", "lug": "Niger-Congo", "ewe": "Niger-Congo",
    "twi": "Niger-Congo", "fon": "Niger-Congo", "wol": "Niger-Congo",
    "ful": "Niger-Congo", "bam": "Niger-Congo", "kon": "Niger-Congo",
    "umb": "Niger-Congo", "tiv": "Niger-Congo", "bem": "Niger-Congo",
    "bnt": "Niger-Congo", "ada": "Niger-Congo", "bin": "Niger-Congo",
    "bam": "Niger-Congo", "fan": "Niger-Congo", "bas": "Niger-Congo",
    "bad": "Niger-Congo", "ven": "Niger-Congo", "ssw": "Niger-Congo",
    "nso": "Niger-Congo", "nbl": "Niger-Congo", "tso": "Niger-Congo",
    "loz": "Niger-Congo", "lun": "Niger-Congo", "kik": "Niger-Congo",
    "kua": "Niger-Congo", "her": "Niger-Congo", "ndo": "Niger-Congo",

    # Austronesian
    "ind": "Austronesian", "msa": "Austronesian", "may": "Austronesian",
    "tgl": "Austronesian", "mao": "Austronesian", "haw": "Austronesian",
    "mlg": "Austronesian", "sun": "Austronesian", "jav": "Austronesian",
    "ceb": "Austronesian", "ilo": "Austronesian", "war": "Austronesian",
    "bug": "Austronesian", "mad": "Austronesian", "min": "Austronesian",
    "ace": "Austronesian", "fij": "Austronesian", "smo": "Austronesian",
    "ton": "Austronesian", "mah": "Austronesian", "pau": "Austronesian",
    "rap": "Austronesian", "tah": "Austronesian", "map": "Austronesian",
    "phi": "Austronesian", "ban": "Austronesian", "bik": "Austronesian",
    "btk": "Austronesian", "bjn": "Austronesian", "jv":  "Austronesian",
    "nia": "Austronesian", "mri": "Austronesian", "tet": "Austronesian",
    "mgm": "Austronesian", "cjm": "Austronesian",

    # Dravidian
    "tam": "Dravidian", "tel": "Dravidian", "kan": "Dravidian",
    "mal": "Dravidian", "gon": "Dravidian", "dra": "Dravidian",
    "bra": "Dravidian", "kur": "Dravidian", "kui": "Dravidian",
    "tcy": "Dravidian", "kfb": "Dravidian",

    # Turkic
    "tur": "Turkic", "uzb": "Turkic", "kaz": "Turkic", "aze": "Turkic",
    "tuk": "Turkic", "kir": "Turkic", "bak": "Turkic", "tat": "Turkic",
    "chv": "Turkic", "uig": "Turkic", "sah": "Turkic", "nog": "Turkic",
    "tyv": "Turkic", "kum": "Turkic", "alt": "Turkic", "crh": "Turkic",
    "trk": "Turkic", "xlc": "Turkic",

    # Japonic
    "jpn": "Japonic",

    # Koreanic
    "kor": "Koreanic",

    # Uralic
    "fin": "Uralic", "hun": "Uralic", "est": "Uralic", "sme": "Uralic",
    "smn": "Uralic", "sms": "Uralic", "smj": "Uralic", "fiu": "Uralic",
    "yrk": "Uralic", "myv": "Uralic", "mdf": "Uralic", "udm": "Uralic",
    "koi": "Uralic", "kpv": "Uralic", "krl": "Uralic", "vep": "Uralic",
    "vot": "Uralic", "liv": "Uralic", "izh": "Uralic", "fkv": "Uralic",
    "niv": "Uralic",

    # Austroasiatic
    "vie": "Austroasiatic", "khm": "Austroasiatic", "mon": "Austroasiatic",
    "mni": "Austroasiatic", "kha": "Austroasiatic", "sat": "Austroasiatic",
    "mun": "Austroasiatic", "aav": "Austroasiatic",

    # Tai-Kadai
    "tha": "Tai-Kadai", "lao": "Tai-Kadai", "zha": "Tai-Kadai",
    "tai": "Tai-Kadai",

    # Mongolic
    "mon": "Mongolic", "bua": "Mongolic", "khk": "Mongolic",

    # Kartvelian
    "kat": "Kartvelian", "geo": "Kartvelian", "lzz": "Kartvelian",
    "xmf": "Kartvelian", "sva": "Kartvelian",

    # Tungusic
    "mnc": "Tungusic", "evn": "Tungusic", "eve": "Tungusic",

    # Hmong-Mien
    "hmn": "Hmong-Mien",

    # Mayan
    "yua": "Mayan", "quc": "Mayan", "mam": "Mayan", "myn": "Mayan",
    "tzo": "Mayan", "tzh": "Mayan",

    # Quechuan / Aymaran
    "que": "Quechuan", "aym": "Aymaran",

    # Nilo-Saharan
    "nub": "Nilo-Saharan", "kau": "Nilo-Saharan", "ssa": "Nilo-Saharan",
    "din": "Nilo-Saharan", "luo": "Nilo-Saharan", "mas": "Nilo-Saharan",

    # Northwest Caucasian
    "abk": "Northwest Caucasian", "ady": "Northwest Caucasian",

    # Northeast Caucasian
    "ava": "Northeast Caucasian", "che": "Northeast Caucasian",
    "lez": "Northeast Caucasian", "inh": "Northeast Caucasian",
    "lak": "Northeast Caucasian", "tab": "Northeast Caucasian",

    # Eskimo-Aleut
    "kal": "Eskimo-Aleut", "iku": "Eskimo-Aleut", "ipk": "Eskimo-Aleut",
    "ale": "Eskimo-Aleut",

    # Na-Dene / Athabaskan
    "nav": "Na-Dene", "ath": "Na-Dene", "den": "Na-Dene",
    "tli": "Na-Dene", "apa": "Na-Dene",

    # Algic
    "cre": "Algic", "oji": "Algic", "alg": "Algic", "mic": "Algic",
    "moh": "Algic", "arp": "Algic", "bla": "Algic",

    # Iroquoian
    "chr": "Iroquoian", "iro": "Iroquoian",

    # Arawakan
    "arw": "Arawakan",

    # Caddoan
    "cad": "Caddoan",

    # Araucanian
    "arn": "Araucanian",

    # Australian
    "aus": "Australian",

    # Trans-New Guinea
    "iri": "Trans-New Guinea",

    # Khoisan
    "khi": "Khoisan",

    # Language isolates
    "eus": "Language Isolate", "baq": "Language Isolate",
    "ain": "Language Isolate", "sux": "Language Isolate",
    "elx": "Language Isolate", "ket": "Language Isolate",
    "yuk": "Language Isolate",

    # Sign languages
    "sgn": "Sign Language", "ase": "Sign Language", "bfi": "Sign Language",
    "dse": "Sign Language", "fsl": "Sign Language", "gsg": "Sign Language",
    "jsl": "Sign Language", "psr": "Sign Language", "bzs": "Sign Language",
    "ins": "Sign Language",

    # Creoles & Pidgins
    "hat": "Creole/Pidgin", "tpi": "Creole/Pidgin", "bis": "Creole/Pidgin",
    "cpe": "Creole/Pidgin", "cpf": "Creole/Pidgin", "cpp": "Creole/Pidgin",
    "crp": "Creole/Pidgin", "pap": "Creole/Pidgin", "acf": "Creole/Pidgin",

    # Constructed
    "epo": "Constructed", "ido": "Constructed", "ina": "Constructed",
    "ile": "Constructed", "vol": "Constructed", "jbo": "Constructed",
    "tlh": "Constructed", "art": "Constructed", "afh": "Constructed",

    "und": "Undetermined",
}

# ── 2. Wikidata SPARQL fallback ───────────────────────────────────────────────
# Maps Wikidata family labels → our top-level family names
WIKIDATA_FAMILY_MAP: dict[str, str] = {
    "Indo-European languages": "Indo-European",
    "Sino-Tibetan languages": "Sino-Tibetan",
    "Afro-Asiatic languages": "Afro-Asiatic",
    "Niger-Congo languages": "Niger-Congo",
    "Austronesian languages": "Austronesian",
    "Dravidian languages": "Dravidian",
    "Turkic languages": "Turkic",
    "Japonic languages": "Japonic",
    "Koreanic languages": "Koreanic",
    "Uralic languages": "Uralic",
    "Austroasiatic languages": "Austroasiatic",
    "Tai-Kadai languages": "Tai-Kadai",
    "Nilo-Saharan languages": "Nilo-Saharan",
    "Mongolic languages": "Mongolic",
    "Kartvelian languages": "Kartvelian",
    "Tungusic languages": "Tungusic",
    "Hmong-Mien languages": "Hmong-Mien",
    "Mayan languages": "Mayan",
    "Quechuan languages": "Quechuan",
    "Aymaran languages": "Aymaran",
    "Na-Dené languages": "Na-Dene",
    "Na-Dene languages": "Na-Dene",
    "Algic languages": "Algic",
    "Iroquoian languages": "Iroquoian",
    "Caddoan languages": "Caddoan",
    "Arawakan languages": "Arawakan",
    "Araucanian languages": "Araucanian",
    "Northwest Caucasian languages": "Northwest Caucasian",
    "Northeast Caucasian languages": "Northeast Caucasian",
    "Eskimo-Aleut languages": "Eskimo-Aleut",
    "Khoisan languages": "Khoisan",
    "Trans–New Guinea languages": "Trans-New Guinea",
    "Australian Aboriginal languages": "Australian",
    "Pama-Nyungan languages": "Australian",
    "sign language": "Sign Language",
    "sign languages": "Sign Language",
    "constructed language": "Constructed",
    "language isolate": "Language Isolate",
    "creole language": "Creole/Pidgin",
    "pidgin language": "Creole/Pidgin",
}

SPARQL_URL = "https://query.wikidata.org/sparql"
SPARQL_HEADERS = {
    "User-Agent": "LingdashFamilyLookup/1.0 (frederico.prado@proton.me)",
    "Accept": "application/sparql-results+json",
}

def query_wikidata(iso_codes: list[str]) -> dict[str, str]:
    """Return {iso_code: family_label} for the given ISO 639-2 codes."""
    if not iso_codes:
        return {}
    values = " ".join(f'"{c}"' for c in iso_codes)
    sparql = f"""
SELECT ?iso (SAMPLE(?familyLabel) AS ?family) WHERE {{
  VALUES ?iso {{ {values} }}
  ?lang wdt:P219 ?iso.
  ?lang wdt:P279+ ?fam.
  ?fam wdt:P31 wd:Q25295.
  FILTER NOT EXISTS {{
    ?lang wdt:P279+ ?parentFam.
    ?parentFam wdt:P31 wd:Q25295.
    ?fam wdt:P279+ ?parentFam.
  }}
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en".
    ?fam rdfs:label ?familyLabel.
  }}
}}
GROUP BY ?iso
"""
    try:
        resp = requests.get(
            SPARQL_URL,
            params={"query": sparql, "format": "json"},
            headers=SPARQL_HEADERS,
            timeout=60,
        )
        resp.raise_for_status()
        bindings = resp.json()["results"]["bindings"]
        return {
            b["iso"]["value"]: b["family"]["value"]
            for b in bindings
            if "iso" in b and "family" in b
        }
    except Exception as e:
        print(f"  Wikidata query failed: {e}")
        return {}


# ── 3. Name-pattern fallback ──────────────────────────────────────────────────
FAMILY_BY_PATTERN: list[tuple[re.Pattern, str]] = [
    (re.compile(r"sign language|signed", re.I),                          "Sign Language"),
    (re.compile(r"creole|pidgin",        re.I),                          "Creole/Pidgin"),
    (re.compile(r"esperanto|interlingua|volapük|klingon|lojban|afrihili", re.I), "Constructed"),
    (re.compile(r"arabic|hebrew|aramaic|amharic|somali|hausa|maltese|tigrinya|afar|cushitic|berber", re.I), "Afro-Asiatic"),
    (re.compile(r"chinese|tibetan|burmese|dzongkha|sino.tibetan", re.I), "Sino-Tibetan"),
    (re.compile(r"turkish|uzbek|kazakh|azerbaijani|uyghur|kyrgyz|tatar|bashkir|turkmen|yakut|chuvash", re.I), "Turkic"),
    (re.compile(r"tamil|telugu|kannada|malayalam|dravidian",             re.I), "Dravidian"),
    (re.compile(r"malay|indonesian|tagalog|javanese|maori|hawaiian|fijian|samoan|austronesian|balinese|bikol|batak", re.I), "Austronesian"),
    (re.compile(r"vietnamese|khmer|cambodian|austroasiatic|munda|santali", re.I), "Austroasiatic"),
    (re.compile(r"\bthai\b|lao\b|zhuang|tai.kadai",                     re.I), "Tai-Kadai"),
    (re.compile(r"finnish|hungarian|estonian|sami|uralic|mordvin|udmurt|komi|mari\b", re.I), "Uralic"),
    (re.compile(r"basque",                                               re.I), "Language Isolate"),
    (re.compile(r"georgian|kartvelian|mingrelian|svan",                  re.I), "Kartvelian"),
    (re.compile(r"mongolian|mongolic|buryat",                            re.I), "Mongolic"),
    (re.compile(r"quechua",                                              re.I), "Quechuan"),
    (re.compile(r"aymara",                                               re.I), "Aymaran"),
    (re.compile(r"mayan|maya\b|yucatec|tzotzil|tzeltal|kiche",          re.I), "Mayan"),
    (re.compile(r"navajo|athabaskan|na.dene|apache",                     re.I), "Na-Dene"),
    (re.compile(r"arapaho|algonquian|algonkian|cree|ojibwe|blackfoot",   re.I), "Algic"),
    (re.compile(r"cherokee|iroquois|mohawk",                             re.I), "Iroquoian"),
    (re.compile(r"swahili|yoruba|igbo|zulu|xhosa|sotho|tswana|shona|kikuyu|bantu|niger.congo|wolof|fulah|bambara", re.I), "Niger-Congo"),
    (re.compile(r"russian|polish|czech|slovak|bulgarian|serbian|croatian|slovene|ukrainian|belarusian|slavic", re.I), "Indo-European"),
    (re.compile(r"french|spanish|italian|portuguese|romanian|catalan|latin|romance|aragonese|asturian|galician|occitan", re.I), "Indo-European"),
    (re.compile(r"english|german|dutch|swedish|norwegian|danish|gothic|germanic|frisian|yiddish|afrikaans", re.I), "Indo-European"),
    (re.compile(r"hindi|urdu|bengali|punjabi|gujarati|marathi|nepali|sinhala|sanskrit|assamese|odia|bihari|indo.aryan|persian|farsi|pashto", re.I), "Indo-European"),
    (re.compile(r"greek|armenian|albanian|celtic|irish|welsh|breton|indo.european|lithuanian|latvian", re.I), "Indo-European"),
    (re.compile(r"japanese",                                             re.I), "Japonic"),
    (re.compile(r"korean",                                               re.I), "Koreanic"),
    (re.compile(r"abkhaz|adyghe|northwest.caucasian",                   re.I), "Northwest Caucasian"),
    (re.compile(r"chechen|ingush|avar|northeast.caucasian|lezgian",     re.I), "Northeast Caucasian"),
    (re.compile(r"inuit|inuktitut|inupiak|aleut|eskimo",                re.I), "Eskimo-Aleut"),
    (re.compile(r"arawak",                                               re.I), "Arawakan"),
    (re.compile(r"mapuche|mapudungun|araucanian",                        re.I), "Araucanian"),
    (re.compile(r"hmong|mien|hmong.mien",                               re.I), "Hmong-Mien"),
    (re.compile(r"nilo.saharan|dinka|nuer|masai",                       re.I), "Nilo-Saharan"),
]


def get_family_from_code_and_name(code: str, name: str) -> str | None:
    c = str(code).strip().lower().split()[0][:3]
    if c in FAMILY_BY_CODE:
        return FAMILY_BY_CODE[c]
    for pattern, family in FAMILY_BY_PATTERN:
        if pattern.search(name):
            return family
    return None


# ── Apply ─────────────────────────────────────────────────────────────────────
path = os.path.join(_DIR, "linglist.csv")
df = pd.read_csv(path)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

df["family"] = df.apply(
    lambda r: get_family_from_code_and_name(r["code"], r["language"]), axis=1
)

unclassified = df[df["family"].isna()]
print(f"After hardcoded lookup: {df['family'].notna().sum()} classified, {len(unclassified)} unclassified")

# ── Wikidata SPARQL for remaining unclassified ────────────────────────────────
if not unclassified.empty:
    iso_codes = [
        str(c).strip().lower().split()[0][:3]
        for c in unclassified["code"].dropna().unique()
    ]
    # Batch in groups of 100 to stay within SPARQL limits
    BATCH = 100
    wd_results: dict[str, str] = {}
    for i in range(0, len(iso_codes), BATCH):
        batch = iso_codes[i : i + BATCH]
        print(f"  Querying Wikidata for batch {i//BATCH + 1} ({len(batch)} codes)…")
        wd_results.update(query_wikidata(batch))
        time.sleep(1.0)  # be polite to Wikidata

    def _apply_wikidata(row):
        if row["family"] is not None:
            return row["family"]
        c = str(row["code"]).strip().lower().split()[0][:3]
        wd_label = wd_results.get(c)
        if wd_label:
            return WIKIDATA_FAMILY_MAP.get(wd_label, wd_label)
        return "Other/Unclassified"

    df["family"] = df.apply(_apply_wikidata, axis=1)
else:
    df["family"] = df["family"].fillna("Other/Unclassified")

print("\nFamily distribution:")
print(df["family"].value_counts().to_string())
print(f"\nStill unclassified: {(df['family'] == 'Other/Unclassified').sum()}")

df.to_csv(path, index=False)
print(f"\nSaved → {path}")
