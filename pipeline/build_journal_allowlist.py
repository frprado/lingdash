"""
build_journal_allowlist.py

Builds pipeline/journal_allowlist.csv — a quality-based journal filter.

A journal QUALIFIES (and its papers are kept) if it meets ANY of:
  • is_in_doaj   — editorially vetted by DOAJ
  • is_core      — in OpenAlex's core literature set
  • h_index ≥ MIN_H_INDEX — has produced sustained cited work

Everything else is marked as not qualified and dropped by the dashboard.
This is country-agnostic: it filters out Sinta-only journals, predatory
journals, and tiny local publications everywhere, not just Indonesia.

Strategy:
  1. Pull all qualifying sources from OpenAlex (3 filtered queries, ~500 pages)
  2. Build a normalised name lookup (O(1) matching, no API calls per journal)
  3. Match every unique journal in data.csv against that lookup
  4. Write journal_allowlist.csv  (columns: journal, qualified, reason)

Usage:
    python pipeline/build_journal_allowlist.py
"""

import os
import re
import time
import requests
import pandas as pd

_DIR    = os.path.dirname(__file__)
# OpenAlex polite pool: set your email so they can contact you if needed.
HEADERS = {"User-Agent": "mailto:YOUR_EMAIL_HERE"}

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_H_INDEX = 15    # journals with h-index below this are considered local/minor


# ── Helpers ───────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    """Normalise a journal name for robust matching."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)   # punctuation → space
    s = re.sub(r"\s+", " ", s)        # collapse whitespace
    return s.strip()


def _fetch_sources(filter_str: str, label: str) -> dict[str, str]:
    """Return {normalised_name: reason} for all sources matching filter_str.

    Uses cursor-based pagination to bypass the 10k page limit.
    """
    lookup: dict[str, str] = {}
    cursor = "*"
    fetched = 0
    while True:
        resp = requests.get(
            "https://api.openalex.org/sources",
            params={"filter": filter_str, "per_page": 200, "cursor": cursor},
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data    = resp.json()
        results = data.get("results", [])
        if not results:
            break
        for src in results:
            name = src.get("display_name") or ""
            if name:
                lookup[_norm(name)] = label
        fetched += len(results)
        total = data["meta"]["count"]
        if fetched % 4000 == 0:
            print(f"    [{label}] {fetched:>6,} / {total:,}")
        cursor = data["meta"].get("next_cursor")
        if not cursor:
            break
        time.sleep(0.12)
    print(f"  ✓ [{label}] {len(lookup):,} sources")
    return lookup


# ── 1. Build the qualified-source lookup ─────────────────────────────────────
print("Fetching qualifying sources from OpenAlex…")
qualified: dict[str, str] = {}

# Order matters: later entries overwrite earlier ones, but all mean "qualified"
qualified.update(_fetch_sources("is_in_doaj:true",                      "doaj"))
qualified.update(_fetch_sources("is_core:true",                          "core"))
qualified.update(_fetch_sources(f"summary_stats.h_index:>{MIN_H_INDEX}", f"h_index>{MIN_H_INDEX}"))

print(f"\nTotal unique qualifying sources: {len(qualified):,}\n")


# ── 2. Load journals from data.csv ────────────────────────────────────────────
df = pd.read_csv(os.path.join(_DIR, "data.csv"))
journals = (
    df["journal"]
    .dropna()
    .loc[lambda s: s.str.strip() != ""]
    .unique()
    .tolist()
)
print(f"Unique journals in dataset: {len(journals):,}")


# ── 3. Match each journal against the lookup ─────────────────────────────────
rows = []
for name in journals:
    key    = _norm(name)
    reason = qualified.get(key)
    rows.append({
        "journal":   name,
        "qualified": reason is not None,
        "reason":    reason or "no_match",
    })

result_df = pd.DataFrame(rows)
n_ok   = result_df["qualified"].sum()
n_drop = (~result_df["qualified"]).sum()
print(f"  Qualified : {n_ok:,}")
print(f"  Dropped   : {n_drop:,}")
print()

# Show reason breakdown for kept journals
print("Qualification breakdown:")
print(result_df[result_df["qualified"]]["reason"].value_counts().to_string())


# ── 4. Save ───────────────────────────────────────────────────────────────────
out_path = os.path.join(_DIR, "journal_allowlist.csv")
result_df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
