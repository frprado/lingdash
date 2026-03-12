"""
export_figures.py

Exports one static PNG per key finding for use in ABOUT.md.
Run from repo root: python pipeline/export_figures.py
"""

import os
import numpy as np
import pandas as pd
import pycountry
import plotly.express as px
import plotly.graph_objects as go

_DIR    = os.path.dirname(__file__)
OUT_DIR = os.path.join(os.path.dirname(_DIR), "assets")
os.makedirs(OUT_DIR, exist_ok=True)

C_PRIMARY = "#6366F1"
C_TEXT    = "#1E293B"
C_MUTED   = "#64748B"
C_BORDER  = "#E2E8F0"
C_BG      = "#F8FAFC"
PALETTE   = ["#6366F1","#10B981","#F59E0B","#EF4444","#3B82F6","#8B5CF6","#06B6D4","#F97316"]

KEEP_TOPICS = {
    "Syntax, Semantics, Linguistic Variation",
    "Linguistic research and analysis",
    "Linguistics and language evolution",
    "Historical Linguistics and Language Studies",
    "Linguistics and Language Analysis",
    "Linguistics and Language Studies",
}

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data…")
_data_path = os.path.join(_DIR, "data.parquet")
if not os.path.exists(_data_path):
    _data_path = os.path.join(_DIR, "data.csv")
df = pd.read_parquet(_data_path) if _data_path.endswith(".parquet") else pd.read_csv(_data_path)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df = df[df["primary_topic"].isin(KEEP_TOPICS)].copy()

_al_path = os.path.join(_DIR, "journal_allowlist.csv")
if os.path.exists(_al_path):
    _al = pd.read_csv(_al_path)
    _qual_col = "qualified" if "qualified" in _al.columns else "linguistics"
    _good = set(_al.loc[_al[_qual_col] == True, "journal"].dropna())
    _has_journal = df["journal"].notna() & (df["journal"].str.strip() != "")
    df = df[_has_journal & df["journal"].isin(_good)].copy()

df["year"]          = pd.to_numeric(df["year"], errors="coerce")
df                  = df.dropna(subset=["year"]).copy()
df["year"]          = df["year"].astype(int)
df["journal"]       = df.get("journal",       pd.Series([""] * len(df))).fillna("")
df["paper_country"] = df.get("paper_country", pd.Series([""] * len(df))).fillna("")

def _country_alpha3(code):
    if not isinstance(code, str) or not code.strip():
        return None
    for lookup in ("alpha_2", "alpha_3"):
        try:
            c = pycountry.countries.get(**{lookup: code.strip().upper()})
            if c and hasattr(c, "alpha_3"):
                return c.alpha_3
        except Exception:
            pass
    return None

def _country_name(code):
    if not isinstance(code, str) or not code.strip():
        return None
    for lookup in ("alpha_2", "alpha_3"):
        try:
            c = pycountry.countries.get(**{lookup: code.strip().upper()})
            if c:
                return c.name
        except Exception:
            pass
    return code

df["country_name"]   = df["paper_country"].apply(_country_name)
df["country_alpha3"] = df["paper_country"].apply(_country_alpha3)

def _parse_langs(x):
    if pd.isna(x):
        return []
    parts = [p.strip() for p in str(x).split(",")]
    return [p.lower().title() for p in parts if p and p.lower() != "unknown"]

df["language_list"] = df.get("language_mentioned", pd.Series([""] * len(df))).apply(_parse_langs)
df_lang = df.explode("language_list").copy()
df_lang["language_list"] = df_lang["language_list"].fillna("")

_linglist_path = os.path.join(_DIR, "linglist.csv")
_linglist = pd.read_csv(_linglist_path)
if "family" in _linglist.columns:
    _lang_family = (
        _linglist[["language", "family"]].dropna()
        .assign(language=lambda d: d["language"].str.strip().str.lower().str.title())
        .drop_duplicates(subset=["language"])
        .set_index("language")["family"].to_dict()
    )
    df_lang["family"] = (
        df_lang["language_list"].str.strip().str.lower().str.title()
        .map(_lang_family).fillna("Other/Unclassified")
    )

print(f"Dataset: {len(df):,} papers, {df['year'].min()}–{df['year'].max()}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _shannon(g):
    c = g.value_counts()
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

def _style(fig, height=400):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        font=dict(family="'Inter', system-ui, sans-serif", size=12, color=C_TEXT),
        colorway=PALETTE,
        margin=dict(l=16, r=16, t=32, b=16),
    )
    return fig

# ── 1. Output growth — publications per year ──────────────────────────────────
print("Generating fig_output_growth.png…")
by_year = df.groupby("year").size().reset_index(name="Papers")
fig1 = px.area(by_year, x="year", y="Papers", line_shape="spline",
               color_discrete_sequence=[C_PRIMARY])
fig1.update_traces(line=dict(width=2, color=C_PRIMARY), fillcolor="rgba(99,102,241,0.12)")
fig1.update_xaxes(title_text="", showgrid=False)
fig1.update_yaxes(title_text="Papers published", gridcolor=C_BORDER)
_style(fig1, height=360)
fig1.write_image(os.path.join(OUT_DIR, "fig_output_growth.png"), width=900, scale=2)

# ── 2. Geographic concentration — stacked area, country share over time ───────
print("Generating fig_geography.png…")
TOP_N = 6
TOP_COUNTRIES = (
    df["country_name"].dropna()
    .loc[lambda s: s.str.strip() != ""]
    .value_counts().head(TOP_N).index.tolist()
)
country_trend_df = (
    df[df["country_name"].isin(TOP_COUNTRIES)]
    .groupby(["year", "country_name"]).size().reset_index(name="count")
)
fig2 = px.area(country_trend_df, x="year", y="count", color="country_name",
               line_shape="spline", color_discrete_sequence=PALETTE,
               category_orders={"country_name": TOP_COUNTRIES})
fig2.update_traces(stackgroup="one", groupnorm="percent", line=dict(width=0.6))
fig2.update_xaxes(title_text="", showgrid=False)
fig2.update_yaxes(title_text="% share of publications", range=[0, 100],
                  ticksuffix="%", gridcolor=C_BORDER)
fig2.update_layout(
    template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    height=400, font=dict(family="'Inter', system-ui, sans-serif", size=12, color=C_TEXT),
    margin=dict(l=16, r=180, t=32, b=16),
    legend=dict(orientation="v", yanchor="middle", y=0.5,
                xanchor="left", x=1.02, bgcolor="rgba(255,255,255,0.9)",
                bordercolor=C_BORDER, borderwidth=1, font=dict(size=11)),
)
fig2.write_image(os.path.join(OUT_DIR, "fig_geography.png"), width=1000, scale=2)

# ── 3. Language bias — treemap by family ─────────────────────────────────────
print("Generating fig_language_bias.png…")
_lang_vc = df_lang[df_lang["language_list"].str.strip() != ""]["language_list"].value_counts()
_EXCLUDE_FAMILIES = {"Other/Unclassified", "Uncoded", "Undetermined"}
_fam = (
    df_lang[
        df_lang["language_list"].str.strip().astype(bool) &
        ~df_lang["family"].isin(_EXCLUDE_FAMILIES)
    ]
    .groupby(["family", "language_list"]).size().reset_index(name="count")
    .rename(columns={"language_list": "language"})
)
fig3 = px.treemap(_fam, path=["family", "language"], values="count",
                  color_discrete_sequence=PALETTE)
fig3.update_traces(
    textinfo="label+percent parent",
    hovertemplate="<b>%{label}</b><br>%{value:,} papers (%{percentParent:.1%} of family)<extra></extra>",
)
fig3.update_layout(
    template="plotly_white", paper_bgcolor="#FFFFFF",
    height=500, showlegend=False,
    font=dict(family="'Inter', system-ui, sans-serif", size=12, color=C_TEXT),
    margin=dict(l=8, r=8, t=32, b=8),
)
fig3.write_image(os.path.join(OUT_DIR, "fig_language_bias.png"), width=1100, scale=2)

# ── 4. Diversity over time — Shannon entropy ──────────────────────────────────
print("Generating fig_diversity.png…")
_lang_rows = df_lang[df_lang["language_list"].str.strip() != ""]
div_df = _lang_rows.groupby("year")["language_list"].apply(_shannon).reset_index(name="entropy")
fig4 = px.line(div_df, x="year", y="entropy", line_shape="spline",
               color_discrete_sequence=[C_PRIMARY])
fig4.update_traces(line=dict(width=2.5, color=C_PRIMARY))
fig4.add_traces(px.area(div_df, x="year", y="entropy", line_shape="spline",
                        color_discrete_sequence=[C_PRIMARY]).data)
fig4.data[-1].update(fillcolor="rgba(99,102,241,0.10)", line=dict(width=0))
fig4.update_xaxes(title_text="", showgrid=False)
fig4.update_yaxes(title_text="Shannon H (bits)", gridcolor=C_BORDER)
_style(fig4, height=360)
fig4.write_image(os.path.join(OUT_DIR, "fig_diversity.png"), width=900, scale=2)

print(f"\nAll figures saved to {OUT_DIR}/")
