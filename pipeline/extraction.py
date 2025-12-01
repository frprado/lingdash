import requests
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter


#Params
url = "https://api.openalex.org/works"

params = {
    "filter": (
        "primary_topic.subfield.id:subfields/1203,"
        "type:types/article,"
        "primary_topic.id:t10034,"
        "publication_year:1960-2024"
    ),
    "sort": "cited_by_count:desc",
    "per_page": 200
}

headers = {
    "User-Agent": "frederico.prado@proton.me"
}

langlist = pd.read_csv("linglist.csv")


#Fetch data
all_results = []
cursor = "*"

while True:
    print("Fetching… cursor:", cursor)
    
    params["cursor"] = cursor
    
    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    
    results = data.get("results", [])
    if not results:
        break
    
    all_results.extend(results)
    
    cursor = data.get("meta", {}).get("next_cursor")
    if not cursor:
        break

print("\nFetched", len(all_results), "works total.")

##Helpers
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

    # build position -> word mapping
    position_word = {}

    for word, positions in inv_map.items():
        for pos in positions:
            position_word[pos] = word

    # sort positions and rejoin
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
    flags=re.IGNORECASE
)

##Build df from json
rows = []

for w in all_results:
    rows.append({
        "title": w.get("title"),
        "year": w.get("publication_year"),
        "doi": w.get("doi"),
        "cited_by": w.get("cited_by_count"),
        "journal": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
        "primary_topic": w.get("primary_topic", {}).get("display_name"),
        "concepts": [c["display_name"] for c in w.get("concepts", [])],
        "top_concept": top_concept(w),
        "paper_country": first_author_country(w),
        "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),

        # AUTHORS COLUMN
        "authors": {
            i+1: a.get("author", {}).get("display_name")
            for i, a in enumerate(w.get("authorships", []))
            if a.get("author", {}).get("display_name")
        }
    })

df = pd.DataFrame(rows)
df.head()

#Save
df.to_csv("data.csv")




