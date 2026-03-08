"""
generate_report.py

Generates a self-contained HTML report with key insights from the
linguistics publications dataset.

Usage:
    python pipeline/generate_report.py
    # opens report.html in your browser
"""

import os
import ast
import datetime
import webbrowser
import pandas as pd
import pycountry
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html

_DIR = os.path.dirname(__file__)

# ── Palette (matches dashboard) ──────────────────────────────────────────────
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

# ── Load & filter (same logic as dashboard) ───────────────────────────────────
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

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["year"]).copy()
df["year"] = df["year"].astype(int)
df["cited_by"] = pd.to_numeric(df.get("cited_by", 0), errors="coerce").fillna(0).astype(int)
df["journal"]       = df.get("journal",       pd.Series([""] * len(df))).fillna("")
df["paper_country"] = df.get("paper_country", pd.Series([""] * len(df))).fillna("")
df["primary_topic"] = df.get("primary_topic", pd.Series([""] * len(df))).fillna("")

_topic_col = "subfield" if "subfield" in df.columns else "primary_topic"


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


df["country_name"]   = df["paper_country"].apply(_country_name)
df["country_alpha3"] = df["paper_country"].apply(_country_alpha3)

# Languages
def _parse_langs(x):
    if pd.isna(x):
        return []
    parts = [p.strip() for p in str(x).split(",")]
    return [p.lower().title() for p in parts if p and p.lower() != "unknown"]

df["language_list"] = df.get("language_mentioned", pd.Series([""] * len(df))).apply(_parse_langs)
df_lang = df.explode("language_list").copy()
df_lang["language_list"] = df_lang["language_list"].fillna("")

# Language family lookup
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
else:
    df_lang["family"] = "Unknown"

print(f"Dataset: {len(df):,} papers, {df['year'].min()}–{df['year'].max()}")


# ── Key stats ─────────────────────────────────────────────────────────────────
total_papers   = len(df)
year_min, year_max = int(df["year"].min()), int(df["year"].max())
total_journals = df["journal"].nunique()
total_countries = df["country_name"].dropna().nunique()

top_country  = df["country_name"].dropna().value_counts().index[0]
top_country_n = int(df["country_name"].value_counts().iloc[0])

_lang_vc = df_lang[df_lang["language_list"].str.strip() != ""]["language_list"].value_counts()
top_lang  = _lang_vc.index[0] if not _lang_vc.empty else "—"
top_lang_n = int(_lang_vc.iloc[0]) if not _lang_vc.empty else 0

top_journal   = df["journal"].value_counts().index[0]
top_journal_n = int(df["journal"].value_counts().iloc[0])

# Growth: compare first 5 years vs last 5 years in dataset
_by_year = df.groupby("year").size()
early = _by_year[_by_year.index <= year_min + 4].mean()
late  = _by_year[_by_year.index >= year_max - 4].mean()
growth_x = round((late / early - 1) * 100) if early > 0 else 0

most_productive_year = int(_by_year.idxmax())
most_productive_n    = int(_by_year.max())


# ── Chart helpers ─────────────────────────────────────────────────────────────
def _style(fig, title=None, height=360):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=10, r=10, t=48 if title else 16, b=10),
        font=dict(family="'Inter', system-ui, sans-serif", size=11, color=C_TEXT),
        title_font=dict(size=13, color=C_TEXT),
        colorway=PALETTE,
        showlegend=False,
        title_text=title,
    )
    return fig


def _chart_html(fig, first=False):
    return to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn" if first else False,
        config={"displayModeBar": False, "responsive": True},
    )


# ── Charts ────────────────────────────────────────────────────────────────────
print("Building charts…")

# 1. Publications per year
by_year = df.groupby("year").size().reset_index(name="Papers")
fig_trend = px.area(by_year, x="year", y="Papers", line_shape="spline",
                    color_discrete_sequence=[C_PRIMARY])
fig_trend.update_traces(line=dict(width=2, color=C_PRIMARY), fillcolor="rgba(99,102,241,0.10)")
fig_trend.update_xaxes(title_text="")
fig_trend.update_yaxes(title_text="Papers published", gridcolor=C_BORDER)
_style(fig_trend, "Publications per year")

# 2. Top 15 countries
country_counts = (
    df["country_name"].dropna()
    .loc[lambda s: s.str.strip() != ""]
    .value_counts().head(15).reset_index()
)
country_counts.columns = ["Country", "Papers"]
colors = [PALETTE[i % len(PALETTE)] for i in range(len(country_counts))]
fig_country_bar = px.bar(country_counts, x="Papers", y="Country", orientation="h")
fig_country_bar.update_traces(marker_color=colors)
fig_country_bar.update_yaxes(autorange="reversed", title_text="")
fig_country_bar.update_xaxes(title_text="Papers")
_style(fig_country_bar, "Top 15 countries by publications", height=400)

# 3. World map
map_df = df[df["country_alpha3"].notna()].groupby("country_alpha3").size().reset_index(name="Papers")
fig_map = px.choropleth(
    map_df, locations="country_alpha3", color="Papers",
    color_continuous_scale=[[0, "#E0E7FF"], [0.4, C_PRIMARY], [1, C_TEXT]],
    projection="natural earth", labels={"Papers": "Papers"},
)
fig_map.update_geos(showframe=False, showcoastlines=True,
                    coastlinecolor=C_BORDER, landcolor="#F8FAFC",
                    oceancolor="#EFF6FF", showocean=True)
_style(fig_map, "Geographic distribution", height=380)
fig_map.update_layout(margin=dict(l=0, r=0, t=44, b=0),
                      coloraxis_colorbar=dict(thickness=10, len=0.6))

# 4. Language family treemap
_fam = (
    df_lang[df_lang["language_list"].str.strip().astype(bool)]
    .groupby(["family", "language_list"]).size().reset_index(name="count")
    .rename(columns={"language_list": "language"})
)
if not _fam.empty:
    fig_treemap = px.treemap(
        _fam, path=["family", "language"], values="count",
        color_discrete_sequence=PALETTE,
    )
    fig_treemap.update_traces(
        textinfo="label+percent parent",
        hovertemplate="<b>%{label}</b><br>%{value:,} papers (%{percentParent:.1%} of family)<extra></extra>",
    )
    _style(fig_treemap, "Languages studied — by family", height=460)
    fig_treemap.update_layout(showlegend=False, margin=dict(l=8, r=8, t=44, b=8))
else:
    fig_treemap = go.Figure()
    _style(fig_treemap, "Languages studied — by family")

# 5. Top languages bar
top_langs = (
    df_lang[df_lang["language_list"].str.strip() != ""]["language_list"]
    .value_counts().head(20).reset_index()
)
top_langs.columns = ["Language", "Papers"]
colors_lang = [PALETTE[i % len(PALETTE)] for i in range(len(top_langs))]
fig_lang_bar = px.bar(top_langs, x="Papers", y="Language", orientation="h")
fig_lang_bar.update_traces(marker_color=colors_lang)
fig_lang_bar.update_yaxes(autorange="reversed", title_text="")
fig_lang_bar.update_xaxes(title_text="Papers mentioning language")
_style(fig_lang_bar, "Most studied languages (top 20)", height=480)

# 6. Top 10 journals
top_journals = df["journal"].value_counts().head(10).reset_index()
top_journals.columns = ["Journal", "Papers"]
colors_j = [PALETTE[i % len(PALETTE)] for i in range(len(top_journals))]
fig_journals = px.bar(top_journals, x="Papers", y="Journal", orientation="h")
fig_journals.update_traces(marker_color=colors_j)
fig_journals.update_yaxes(autorange="reversed", title_text="")
fig_journals.update_xaxes(title_text="Papers")
_style(fig_journals, "Top 10 journals by volume", height=360)

# 7. Subfield / topic distribution
topic_counts = (
    df[_topic_col].dropna()
    .loc[lambda s: s.str.strip() != ""]
    .value_counts().reset_index()
)
topic_counts.columns = ["Subfield", "Papers"]
colors_t = [PALETTE[i % len(PALETTE)] for i in range(len(topic_counts))]
fig_topics = px.bar(topic_counts, x="Papers", y="Subfield", orientation="h")
fig_topics.update_traces(marker_color=colors_t)
fig_topics.update_yaxes(autorange="reversed", title_text="")
fig_topics.update_xaxes(title_text="Papers")
_style(fig_topics, "Papers by subfield", height=max(280, len(topic_counts) * 36 + 60))

# 8. Language share over time (top 6)
TOP_LANGS = _lang_vc.head(6).index.tolist()
lang_trend_df = (
    df_lang[df_lang["language_list"].isin(TOP_LANGS)]
    .groupby(["year", "language_list"]).size().reset_index(name="count")
)
if not lang_trend_df.empty:
    fig_lang_trend = px.area(
        lang_trend_df, x="year", y="count", color="language_list",
        color_discrete_sequence=PALETTE, line_shape="spline",
        category_orders={"language_list": TOP_LANGS},
        labels={"language_list": "Language", "count": "Papers", "year": ""},
    )
    fig_lang_trend.update_traces(stackgroup="one", groupnorm="percent", line=dict(width=0.6))
    fig_lang_trend.update_yaxes(title_text="% share", range=[0, 100],
                                ticksuffix="%", gridcolor=C_BORDER)
    _style(fig_lang_trend, "Language share over time (top 6)", height=360)
    fig_lang_trend.update_layout(
        showlegend=True,
        margin=dict(l=10, r=140, t=44, b=10),
        legend=dict(orientation="v", yanchor="middle", y=0.5,
                    xanchor="left", x=1.02,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor=C_BORDER,
                    borderwidth=1, font=dict(size=10)),
    )
else:
    fig_lang_trend = go.Figure()
    _style(fig_lang_trend, "Language share over time")


# ── Render HTML ───────────────────────────────────────────────────────────────
print("Rendering HTML…")

chart_trend       = _chart_html(fig_trend,       first=True)
chart_country_bar = _chart_html(fig_country_bar)
chart_map         = _chart_html(fig_map)
chart_treemap     = _chart_html(fig_treemap)
chart_lang_bar    = _chart_html(fig_lang_bar)
chart_journals    = _chart_html(fig_journals)
chart_topics      = _chart_html(fig_topics)
chart_lang_trend  = _chart_html(fig_lang_trend)

generated = datetime.date.today().strftime("%B %d, %Y")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Linguistics Research — Key Insights Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Inter', system-ui, sans-serif;
    background: {C_BG};
    color: {C_TEXT};
    line-height: 1.6;
    font-size: 14px;
  }}

  /* ── Page header ── */
  .page-header {{
    background: {C_TEXT};
    color: #fff;
    padding: 52px 64px 44px;
  }}
  .page-header .eyebrow {{
    font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: {C_PRIMARY}; margin-bottom: 12px;
  }}
  .page-header h1 {{
    font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em;
    line-height: 1.15; margin-bottom: 10px;
  }}
  .page-header .subtitle {{
    font-size: 13px; color: #94A3B8; margin-bottom: 28px;
  }}
  .page-header .meta {{
    font-size: 11px; color: #64748B;
  }}

  /* ── KPI row ── */
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    background: {C_TEXT};
    border-top: 1px solid #334155;
  }}
  .kpi {{
    padding: 24px 32px;
    border-right: 1px solid #334155;
  }}
  .kpi:last-child {{ border-right: none; }}
  .kpi-label {{
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #64748B; margin-bottom: 6px;
  }}
  .kpi-value {{
    font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em;
    color: #fff; line-height: 1.1;
  }}
  .kpi-sub {{
    font-size: 10px; color: #94A3B8; margin-top: 4px;
  }}
  .kpi-accent-indigo  {{ border-top: 3px solid {C_PRIMARY}; }}
  .kpi-accent-emerald {{ border-top: 3px solid #10B981; }}
  .kpi-accent-amber   {{ border-top: 3px solid #F59E0B; }}
  .kpi-accent-rose    {{ border-top: 3px solid #EF4444; }}

  /* ── Content ── */
  .content {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 52px 32px 80px;
  }}

  /* ── Section ── */
  .section {{ margin-bottom: 56px; }}
  .section-title {{
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: {C_MUTED}; margin-bottom: 6px;
  }}
  .section-heading {{
    font-size: 1.3rem; font-weight: 700; color: {C_TEXT};
    letter-spacing: -0.02em; margin-bottom: 8px;
  }}
  .section-body {{
    font-size: 13px; color: {C_MUTED}; margin-bottom: 20px;
    max-width: 700px; line-height: 1.65;
  }}
  .section-divider {{
    border: none; border-top: 1px solid {C_BORDER};
    margin-bottom: 28px;
  }}

  /* ── Chart cards ── */
  .card {{
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    overflow: hidden;
    margin-bottom: 20px;
  }}
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }}
  .three-col {{
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 20px;
  }}

  /* ── Insight callout ── */
  .insights {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 28px;
  }}
  .insight {{
    background: #fff;
    border-radius: 10px;
    padding: 18px 20px;
    border-left: 3px solid {C_PRIMARY};
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    font-size: 12px;
    color: {C_TEXT};
    line-height: 1.55;
  }}
  .insight strong {{
    display: block;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {C_MUTED};
    margin-bottom: 6px;
  }}
  .insight.emerald {{ border-color: #10B981; }}
  .insight.amber   {{ border-color: #F59E0B; }}
  .insight.rose    {{ border-color: #EF4444; }}

  /* ── Footer ── */
  .footer {{
    text-align: center;
    padding: 32px;
    font-size: 11px;
    color: {C_MUTED};
    border-top: 1px solid {C_BORDER};
  }}

  @media print {{
    body {{ background: #fff; }}
    .page-header {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .kpi-row {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .card {{ box-shadow: none; border: 1px solid {C_BORDER}; page-break-inside: avoid; }}
    .insight {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>

<!-- ── Header ── -->
<div class="page-header">
  <div class="eyebrow">Research Intelligence Report</div>
  <h1>Linguistics Research<br>Global Overview</h1>
  <div class="subtitle">
    Based on {total_papers:,} peer-reviewed articles indexed in OpenAlex &middot;
    {year_min}&ndash;{year_max} &middot; {total_countries} countries &middot; {total_journals:,} journals
  </div>
  <div class="meta">Generated {generated}</div>
</div>

<!-- ── KPI row ── -->
<div class="kpi-row">
  <div class="kpi kpi-accent-indigo">
    <div class="kpi-label">Total publications</div>
    <div class="kpi-value">{total_papers:,}</div>
    <div class="kpi-sub">{year_min}&ndash;{year_max}</div>
  </div>
  <div class="kpi kpi-accent-emerald">
    <div class="kpi-label">Top producing country</div>
    <div class="kpi-value">{top_country}</div>
    <div class="kpi-sub">{top_country_n:,} papers</div>
  </div>
  <div class="kpi kpi-accent-amber">
    <div class="kpi-label">Most studied language</div>
    <div class="kpi-value">{top_lang}</div>
    <div class="kpi-sub">{top_lang_n:,} papers</div>
  </div>
  <div class="kpi kpi-accent-rose">
    <div class="kpi-label">Peak year</div>
    <div class="kpi-value">{most_productive_year}</div>
    <div class="kpi-sub">{most_productive_n:,} papers</div>
  </div>
</div>

<!-- ── Content ── -->
<div class="content">

  <!-- 1. Growth -->
  <div class="section">
    <div class="section-title">Section 01</div>
    <div class="section-heading">Publication growth over time</div>
    <div class="section-body">
      Linguistics research output has grown substantially since the 1960s.
      Comparing the first five years of the dataset ({year_min}&ndash;{year_min+4})
      with the most recent five ({year_max-4}&ndash;{year_max}), annual output
      increased by approximately <strong>{growth_x:+,}%</strong>.
      The peak year was {most_productive_year} with {most_productive_n:,} publications.
    </div>
    <hr class="section-divider">
    <div class="card">{chart_trend}</div>
  </div>

  <!-- 2. Geography -->
  <div class="section">
    <div class="section-title">Section 02</div>
    <div class="section-heading">Geographic distribution</div>
    <div class="section-body">
      Research output is concentrated in a handful of countries, with
      {top_country} leading at {top_country_n:,} papers ({top_country_n / total_papers:.0%} of
      the corpus). The dataset covers {total_countries} distinct countries.
    </div>
    <hr class="section-divider">
    <div class="card">{chart_map}</div>
    <div class="card">{chart_country_bar}</div>
  </div>

  <!-- 3. Languages -->
  <div class="section">
    <div class="section-title">Section 03</div>
    <div class="section-heading">Languages studied</div>
    <div class="section-body">
      The treemap below shows which language families — and individual
      languages within them — receive the most attention in the literature.
      {top_lang} is the most studied language, appearing in {top_lang_n:,} papers.
      The share-over-time chart reveals how research attention has shifted
      between languages across decades.
    </div>
    <hr class="section-divider">
    <div class="card">{chart_treemap}</div>
    <div class="two-col">
      <div class="card">{chart_lang_bar}</div>
      <div class="card">{chart_lang_trend}</div>
    </div>
  </div>

  <!-- 4. Journals -->
  <div class="section">
    <div class="section-title">Section 04</div>
    <div class="section-heading">Journals</div>
    <div class="section-body">
      The corpus covers {total_journals:,} qualified journals (filtered by DOAJ
      membership, OpenAlex core status, or h-index &gt; 15).
      {top_journal} is the most productive single journal with {top_journal_n:,} papers.
    </div>
    <hr class="section-divider">
    <div class="card">{chart_journals}</div>
  </div>

  <!-- 5. Subfields -->
  <div class="section">
    <div class="section-title">Section 05</div>
    <div class="section-heading">Subfields</div>
    <div class="section-body">
      Papers are classified into linguistics subfields
      {"using zero-shot embedding similarity" if _topic_col == "subfield" else "by their OpenAlex primary topic"}.
    </div>
    <hr class="section-divider">
    <div class="card">{chart_topics}</div>
  </div>

</div><!-- /content -->

<div class="footer">
  Data source: OpenAlex &middot; Filtered to peer-reviewed journal articles in linguistics topics &middot;
  Journal quality: DOAJ + OpenAlex core + h-index &gt; 15 &middot;
  Generated {generated}
</div>

</body>
</html>"""

out_path = os.path.join(_DIR, "report.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nReport saved → {out_path}")
webbrowser.open(f"file://{os.path.abspath(out_path)}")
