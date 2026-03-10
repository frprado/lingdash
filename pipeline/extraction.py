import os
import requests
import pandas as pd
import re


# ─────────────────────────────────────────
# Target topics (primary_topic.id filter)
# ─────────────────────────────────────────
TOPIC_IDS = [
    "T10034",   # Syntax, Semantics, Linguistic Variation
    "T12373",   # Linguistic research and analysis
    "T13156",   # Historical Linguistics and Language Studies
    "T13538",   # Linguistics and Language Analysis
    "T13260",   # Linguistics and Language Studies
]

URL = "https://api.openalex.org/works"

params = {
    "filter": (
        "primary_topic.id:" + "|".join(TOPIC_IDS) + ","
        "type:types/article,"
        "publication_year:1960-2026"
    ),
    "sort": "cited_by_count:desc",
    "per_page": 200,
}

# OpenAlex polite pool: set your email so they can contact you if needed.
# See https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication
headers = {
    "User-Agent": "mailto:YOUR_EMAIL_HERE"
}

_DIR = os.path.dirname(__file__)
langlist = pd.read_csv(os.path.join(_DIR, "linglist.csv"))


# ─────────────────────────────────────────
# Fetch (cursor-based pagination)
# ─────────────────────────────────────────
all_results = []
cursor = "*"
page = 0

while True:
    page += 1
    params["cursor"] = cursor

    resp = requests.get(URL, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    if not results:
        break

    all_results.extend(results)

    total = data.get("meta", {}).get("count", "?")
    print(f"Page {page:>4} | fetched {len(all_results):>7} / {total}")

    cursor = data.get("meta", {}).get("next_cursor")
    if not cursor:
        break

print(f"\nDone — {len(all_results)} works total.")


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def top_concept(work):
    concepts = work.get("concepts", [])
    if not concepts:
        return None
    return max(concepts, key=lambda c: c["score"])["display_name"]


def first_author_country(work):
    for auth in work.get("authorships", []):
        for inst in auth.get("institutions", []):
            if inst.get("country_code"):
                return inst["country_code"]
    return None


def reconstruct_abstract(inv_map):
    if not inv_map:
        return None
    position_word = {}
    for word, positions in inv_map.items():
        for pos in positions:
            position_word[pos] = word
    words = [position_word[i] for i in sorted(position_word)]
    return " ".join(words)


languages = (
    langlist["language"]
    .dropna()
    .drop_duplicates()
    .str.strip()
    .tolist()
)

lang_pattern = re.compile(
    r"\b(" + "|".join(re.escape(lang) for lang in languages) + r")\b",
    flags=re.IGNORECASE,
)


# ─────────────────────────────────────────
# Build DataFrame
# ─────────────────────────────────────────
def _journal_fields(work):
    """Return (display_name, source_id) only when the source is a journal."""
    source = (work.get("primary_location") or {}).get("source") or {}
    if source.get("type") == "journal":
        return source.get("display_name"), source.get("id")
    return None, None


rows = []

for w in all_results:
    title    = w.get("title") or ""
    abstract = reconstruct_abstract(w.get("abstract_inverted_index")) or ""
    text     = f"{title} {abstract}"

    found_langs = lang_pattern.findall(text)
    seen, unique_langs = set(), []
    for lang in found_langs:
        key = lang.lower()
        if key not in seen:
            seen.add(key)
            unique_langs.append(lang.title())
    language_mentioned = ", ".join(unique_langs) if unique_langs else None

    rows.append({
        "title":              w.get("title"),
        "year":               w.get("publication_year"),
        "doi":                w.get("doi"),
        "cited_by":           w.get("cited_by_count"),
        "journal":            _journal_fields(w)[0],
        "source_id":          _journal_fields(w)[1],
        "primary_topic":      (w.get("primary_topic") or {}).get("display_name"),
        "concepts":           [c["display_name"] for c in w.get("concepts", [])],
        "top_concept":        top_concept(w),
        "paper_country":      first_author_country(w),
        "abstract":           abstract or None,
        "language_mentioned": language_mentioned,
        "authors": {
            i + 1: a.get("author", {}).get("display_name")
            for i, a in enumerate(w.get("authorships", []))
            if a.get("author", {}).get("display_name")
        },
    })

df = pd.DataFrame(rows)

# ─────────────────────────────────────────
# Save
# ─────────────────────────────────────────
out_path = os.path.join(_DIR, "data.csv")
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows → {out_path}")
