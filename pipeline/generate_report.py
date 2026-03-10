"""
generate_report.py

Generates a self-contained HTML report with key insights from the
linguistics publications dataset.

Usage:
    python pipeline/generate_report.py            # HTML only (opens in browser)
    python pipeline/generate_report.py --pdf      # also export report.pdf
    python pipeline/generate_report.py --png      # also export report.png
    python pipeline/generate_report.py --pdf --png

Requires for PDF/PNG export:
    pip install playwright
    playwright install chromium
"""

import argparse
import os
import webbrowser
import numpy as np
import pandas as pd
import pycountry
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html

# ── CLI args ───────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Generate linguistics report")
_parser.add_argument("--pdf", action="store_true", help="Export report.pdf via Playwright")
_parser.add_argument("--png", action="store_true", help="Export report.png via Playwright")
_args = _parser.parse_args()

_DIR = os.path.dirname(__file__)

# ── Palette ───────────────────────────────────────────────────────────────────
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

# ── Load & filter ─────────────────────────────────────────────────────────────
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

df["year"]         = pd.to_numeric(df["year"], errors="coerce")
df                 = df.dropna(subset=["year"]).copy()
df["year"]         = df["year"].astype(int)
df["cited_by"]     = pd.to_numeric(df.get("cited_by", 0), errors="coerce").fillna(0).astype(int)
df["journal"]      = df.get("journal",       pd.Series([""] * len(df))).fillna("")
df["paper_country"]= df.get("paper_country", pd.Series([""] * len(df))).fillna("")
df["primary_topic"]= df.get("primary_topic", pd.Series([""] * len(df))).fillna("")

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
else:
    df_lang["family"] = "Unknown"

print(f"Dataset: {len(df):,} papers, {df['year'].min()}–{df['year'].max()}")

# ── Key stats ─────────────────────────────────────────────────────────────────
total_papers    = len(df)
year_min, year_max = int(df["year"].min()), int(df["year"].max())
total_journals  = df["journal"].nunique()
total_countries = df["country_name"].dropna().nunique()

top_country   = df["country_name"].dropna().value_counts().index[0]
top_country_n = int(df["country_name"].value_counts().iloc[0])

_lang_vc   = df_lang[df_lang["language_list"].str.strip() != ""]["language_list"].value_counts()
top_lang   = _lang_vc.index[0] if not _lang_vc.empty else "—"
top_lang_n = int(_lang_vc.iloc[0]) if not _lang_vc.empty else 0

top_journal   = df["journal"].value_counts().index[0]
top_journal_n = int(df["journal"].value_counts().iloc[0])

_by_year = df.groupby("year").size()
early    = _by_year[_by_year.index <= year_min + 4].mean()
late     = _by_year[_by_year.index >= year_max - 4].mean()
growth_x = round((late / early - 1) * 100) if early > 0 else 0

most_productive_year = int(_by_year.idxmax())
most_productive_n    = int(_by_year.max())


# ── Chart helpers ─────────────────────────────────────────────────────────────
def _shannon(g):
    c = g.value_counts()
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def _style(fig, title=None, height=360, legend=False):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=10, r=150 if legend else 10, t=48 if title else 16, b=10),
        font=dict(family="'Inter', system-ui, sans-serif", size=11, color=C_TEXT),
        title_font=dict(size=13, color=C_TEXT),
        colorway=PALETTE,
        showlegend=legend,
        title_text=title,
    )
    if legend:
        fig.update_layout(legend=dict(
            orientation="v", yanchor="middle", y=0.5,
            xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.9)", bordercolor=C_BORDER,
            borderwidth=1, font=dict(size=10),
        ))
    return fig


def _norm_area(df_in, x, y, color, title, cat_order, height=360):
    fig = px.area(df_in, x=x, y=y, color=color,
                  color_discrete_sequence=PALETTE, line_shape="spline",
                  category_orders={color: cat_order})
    fig.update_traces(stackgroup="one", groupnorm="percent", line=dict(width=0.6))
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="% share", range=[0, 100],
                     ticksuffix="%", gridcolor=C_BORDER)
    _style(fig, title, height=height, legend=True)
    return fig


def _chart(fig, first=False):
    return to_html(fig, full_html=False,
                   include_plotlyjs="cdn" if first else False,
                   config={"displayModeBar": False, "responsive": True})


# ── Charts ────────────────────────────────────────────────────────────────────
print("Building charts…")

H = 380  # shared chart height

# 1. Publications per year
by_year = df.groupby("year").size().reset_index(name="Papers")
fig_trend = px.area(by_year, x="year", y="Papers", line_shape="spline",
                    color_discrete_sequence=[C_PRIMARY])
fig_trend.update_traces(line=dict(width=2, color=C_PRIMARY), fillcolor="rgba(99,102,241,0.10)")
fig_trend.update_xaxes(title_text="")
fig_trend.update_yaxes(title_text="Papers published", gridcolor=C_BORDER)
_style(fig_trend, "Publications per year", height=260)

# 2. World map
map_df = df[df["country_alpha3"].notna()].groupby("country_alpha3").size().reset_index(name="Papers")
fig_map = px.choropleth(
    map_df, locations="country_alpha3", color="Papers",
    color_continuous_scale=[[0, "#E0E7FF"], [0.4, C_PRIMARY], [1, C_TEXT]],
    projection="natural earth",
)
fig_map.update_geos(showframe=False, showcoastlines=True,
                    coastlinecolor=C_BORDER, landcolor="#F8FAFC",
                    oceancolor="#EFF6FF", showocean=True)
_style(fig_map, "Geographic distribution", height=H)
fig_map.update_layout(margin=dict(l=0, r=0, t=44, b=0),
                      coloraxis_colorbar=dict(thickness=10, len=0.6))

# 3. Country share over time (top 6)
TOP_COUNTRIES = (
    df["country_name"].dropna()
    .loc[lambda s: s.str.strip() != ""]
    .value_counts().head(6).index.tolist()
)
country_trend_df = (
    df[df["country_name"].isin(TOP_COUNTRIES)]
    .groupby(["year", "country_name"]).size().reset_index(name="count")
)
fig_country_trend = _norm_area(
    country_trend_df, "year", "count", "country_name",
    "Country share over time (top 6)",
    TOP_COUNTRIES, height=H,
)

# 4. Language family treemap
_EXCLUDE_FAMILIES = {"Other/Unclassified", "Uncoded", "Undetermined"}
_fam = (
    df_lang[
        df_lang["language_list"].str.strip().astype(bool) &
        ~df_lang["family"].isin(_EXCLUDE_FAMILIES)
    ]
    .groupby(["family", "language_list"]).size().reset_index(name="count")
    .rename(columns={"language_list": "language"})
)
if not _fam.empty:
    fig_treemap = px.treemap(_fam, path=["family", "language"], values="count",
                             color_discrete_sequence=PALETTE)
    fig_treemap.update_traces(
        textinfo="label+percent parent",
        hovertemplate="<b>%{label}</b><br>%{value:,} papers (%{percentParent:.1%} of family)<extra></extra>",
    )
    _style(fig_treemap, "Languages studied — by family", height=H)
    fig_treemap.update_layout(showlegend=False, margin=dict(l=8, r=8, t=44, b=8))
else:
    fig_treemap = go.Figure()
    _style(fig_treemap, "Languages studied — by family", height=H)

# 5. Language share over time (top 6)
TOP_LANGS = _lang_vc.head(6).index.tolist()
lang_trend_df = (
    df_lang[df_lang["language_list"].isin(TOP_LANGS)]
    .groupby(["year", "language_list"]).size().reset_index(name="count")
)
if not lang_trend_df.empty:
    fig_lang_trend = _norm_area(
        lang_trend_df, "year", "count", "language_list",
        "Language share over time (top 6)", TOP_LANGS, height=H,
    )
else:
    fig_lang_trend = go.Figure()
    _style(fig_lang_trend, "Language share over time", height=H)

# 6. Language diversity index per year (Shannon entropy)
_lang_rows = df_lang[df_lang["language_list"].str.strip() != ""]
div_df = _lang_rows.groupby("year")["language_list"].apply(_shannon).reset_index(name="entropy")
fig_diversity = px.line(div_df, x="year", y="entropy", line_shape="spline",
                        color_discrete_sequence=[C_PRIMARY])
fig_diversity.update_traces(line=dict(width=2, color=C_PRIMARY))
fig_diversity.update_xaxes(title_text="")
fig_diversity.update_yaxes(title_text="Shannon H (bits)", gridcolor=C_BORDER)
_style(fig_diversity, "Language diversity index per year", height=H)

# 7. Language diversity per journal (top 14 by volume)
_TOP_J = df["journal"].value_counts().head(14).index.tolist()
_jdf = df_lang[
    df_lang["journal"].isin(_TOP_J) &
    df_lang["language_list"].str.strip().astype(bool)
]
_journal_div = (
    _jdf.groupby("journal")["language_list"]
    .apply(_shannon)
    .reset_index(name="entropy")
    .sort_values("entropy")
)
# Shorten long journal names for display
_journal_div["label"] = _journal_div["journal"].apply(
    lambda s: s if len(s) <= 38 else s[:36] + "…"
)
fig_journal_div = px.bar(
    _journal_div, x="entropy", y="label", orientation="h",
    color="entropy", color_continuous_scale=[[0, "#E0E7FF"], [1, C_PRIMARY]],
)
fig_journal_div.update_coloraxes(showscale=False)
fig_journal_div.update_yaxes(title_text="", tickfont=dict(size=10))
fig_journal_div.update_xaxes(title_text="Shannon H (bits)", gridcolor=C_BORDER)
_style(fig_journal_div, "Language diversity by journal", height=H)


# ── Render HTML ───────────────────────────────────────────────────────────────
print("Rendering HTML…")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Linguistics Research — Global Overview</title>
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

  /* ── Header ── */
  .page-header {{
    background: {C_TEXT};
    color: #fff;
    padding: 44px 64px 36px;
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: end;
    gap: 40px;
  }}
  .eyebrow {{
    font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: {C_PRIMARY}; margin-bottom: 10px;
  }}
  .page-header h1 {{
    font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em;
    line-height: 1.15; margin-bottom: 8px;
  }}
  .page-header .subtitle {{
    font-size: 12px; color: #94A3B8;
  }}
  .header-intro {{
    max-width: 420px; font-size: 12px; color: #94A3B8; line-height: 1.7;
    border-left: 2px solid #334155; padding-left: 20px;
  }}
  .header-intro strong {{ color: #CBD5E1; font-weight: 500; }}

  /* ── KPI row ── */
  .kpi-row {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    background: {C_TEXT}; border-top: 1px solid #334155;
  }}
  .kpi {{
    padding: 20px 28px; border-right: 1px solid #334155;
  }}
  .kpi:last-child {{ border-right: none; }}
  .kpi-label {{
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #64748B; margin-bottom: 5px;
  }}
  .kpi-value {{
    font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em;
    color: #fff; line-height: 1.1;
  }}
  .kpi-sub {{ font-size: 10px; color: #94A3B8; margin-top: 3px; }}
  .kpi-accent-indigo  {{ border-top: 3px solid {C_PRIMARY}; }}
  .kpi-accent-emerald {{ border-top: 3px solid #10B981; }}
  .kpi-accent-amber   {{ border-top: 3px solid #F59E0B; }}
  .kpi-accent-rose    {{ border-top: 3px solid #EF4444; }}

  /* ── Content ── */
  .content {{
    max-width: 1400px; margin: 0 auto; padding: 40px 40px 64px;
  }}

  /* ── Section ── */
  .section {{ margin-bottom: 48px; }}
  .section-header {{
    display: grid; grid-template-columns: auto 1fr;
    gap: 0 32px; align-items: start;
    margin-bottom: 20px; padding-bottom: 16px;
    border-bottom: 1px solid {C_BORDER};
  }}
  .section-tag {{
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: {C_PRIMARY};
    background: #EEF2FF; padding: 4px 10px; border-radius: 4px;
    white-space: nowrap; margin-top: 3px;
  }}
  .section-heading {{
    font-size: 1.05rem; font-weight: 700; color: {C_TEXT};
    letter-spacing: -0.01em; margin-bottom: 5px;
  }}
  .section-body {{
    font-size: 12px; color: {C_MUTED}; line-height: 1.75; max-width: 900px;
  }}
  .section-body strong {{ color: {C_TEXT}; font-weight: 600; }}

  /* ── Cards & grids ── */
  .card {{
    background: #fff; border-radius: 10px; overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
  }}
  .two-col {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
  }}
  .col-5-5 {{
    display: grid; grid-template-columns: 5fr 5fr; gap: 16px;
  }}
  .col-6-4 {{
    display: grid; grid-template-columns: 6fr 4fr; gap: 16px;
  }}

  /* ── Footer ── */
  .footer {{
    text-align: center; padding: 24px;
    font-size: 11px; color: {C_MUTED};
    border-top: 1px solid {C_BORDER};
  }}

  @media print {{
    body {{ background: #fff; }}
    .page-header, .kpi-row {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .card {{ box-shadow: none; border: 1px solid {C_BORDER}; page-break-inside: avoid; }}
    .two-col, .col-5-5, .col-6-4 {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>

<!-- ── Header ── -->
<div class="page-header">
  <div>
    <div class="eyebrow">Research Overview · OpenAlex</div>
    <h1>Linguistics Research<br>Global Overview</h1>
    <div class="subtitle">
      {total_papers:,} peer-reviewed articles &middot; {year_min}&ndash;{year_max} &middot;
      {total_countries} countries &middot; {total_journals:,} journals
    </div>
  </div>
  <div class="header-intro">
    <strong>Which languages get studied</strong> is not a neutral question.
    It reflects patterns of academic prestige, institutional funding, and colonial history.
    This report maps where linguistics research comes from, which languages it focuses on,
    and how the breadth of the field's agenda has changed over {year_max - year_min} years.
  </div>
</div>

<!-- ── KPI strip ── -->
<div class="kpi-row">
  <div class="kpi kpi-accent-indigo">
    <div class="kpi-label">Total publications</div>
    <div class="kpi-value">{total_papers:,}</div>
    <div class="kpi-sub">{year_min}&ndash;{year_max}</div>
  </div>
  <div class="kpi kpi-accent-emerald">
    <div class="kpi-label">Top producing country</div>
    <div class="kpi-value">{top_country}</div>
    <div class="kpi-sub">{top_country_n:,} papers ({top_country_n / total_papers:.0%} of corpus)</div>
  </div>
  <div class="kpi kpi-accent-amber">
    <div class="kpi-label">Most studied language</div>
    <div class="kpi-value">{top_lang}</div>
    <div class="kpi-sub">{top_lang_n:,} papers</div>
  </div>
  <div class="kpi kpi-accent-rose">
    <div class="kpi-label">Output growth</div>
    <div class="kpi-value">{growth_x:+,}%</div>
    <div class="kpi-sub">first 5 yrs → last 5 yrs</div>
  </div>
</div>

<!-- ── Content ── -->
<div class="content">

  <!-- 1. Growth -->
  <div class="section">
    <div class="section-header">
      <div class="section-tag">01 · Output</div>
      <div>
        <div class="section-heading">Publication growth over time</div>
        <div class="section-body">
          Annual output grew by approximately <strong>{growth_x:+,}%</strong> comparing the first five years
          ({year_min}–{year_min + 4}) to the most recent five ({year_max - 4}–{year_max}).
          The peak year on record is <strong>{most_productive_year}</strong> ({most_productive_n:,} publications).
          Growth reflects both a genuine expansion of the field and the broader digitisation of scholarly
          publishing that has made more work indexable.
        </div>
      </div>
    </div>
    <div class="card">{_chart(fig_trend, first=True)}</div>
  </div>

  <!-- 2. Geography -->
  <div class="section">
    <div class="section-header">
      <div class="section-tag">02 · Geography</div>
      <div>
        <div class="section-heading">Where research comes from</div>
        <div class="section-body">
          Research output is geographically concentrated: <strong>{top_country}</strong> alone accounts for
          {top_country_n:,} papers — {top_country_n / total_papers:.0%} of the entire corpus.
          The stacked-area chart shows how the relative share of the top six producing countries has
          shifted over time. A declining share for any country does not necessarily mean fewer papers —
          it often reflects growth elsewhere.
        </div>
      </div>
    </div>
    <div class="col-6-4">
      <div class="card">{_chart(fig_map)}</div>
      <div class="card">{_chart(fig_country_trend)}</div>
    </div>
  </div>

  <!-- 3. Languages -->
  <div class="section">
    <div class="section-header">
      <div class="section-tag">03 · Languages</div>
      <div>
        <div class="section-heading">Which languages are studied</div>
        <div class="section-body">
          The treemap groups languages by family; each tile's size reflects the number of papers
          mentioning that language. <strong>{top_lang}</strong> is by far the most studied ({top_lang_n:,} papers).
          The share chart reveals how attention has shifted among the top six languages over the decades —
          a narrowing or widening spread indicates changing research priorities within the field.
        </div>
      </div>
    </div>
    <div class="two-col">
      <div class="card">{_chart(fig_treemap)}</div>
      <div class="card">{_chart(fig_lang_trend)}</div>
    </div>
  </div>

  <!-- 4. Diversity -->
  <div class="section">
    <div class="section-header">
      <div class="section-tag">04 · Diversity</div>
      <div>
        <div class="section-heading">How broad is the field's linguistic gaze?</div>
        <div class="section-body">
          Shannon entropy (H) measures how evenly attention is spread across all languages studied in a given
          year or journal. <strong>Higher H = more languages studied in roughly equal proportions;</strong>
          lower H = research concentrated on a few languages. The left chart tracks this over time;
          the right shows which journals publish the broadest range of language research — a proxy for
          editorial openness to linguistic diversity.
        </div>
      </div>
    </div>
    <div class="col-5-5">
      <div class="card">{_chart(fig_diversity)}</div>
      <div class="card">{_chart(fig_journal_div)}</div>
    </div>
  </div>

</div><!-- /content -->

<div class="footer">
  Data: OpenAlex &middot; Peer-reviewed journal articles &middot;
  Journal quality filter: DOAJ + OpenAlex core + h-index &gt;&nbsp;15 &middot;
  {year_min}&ndash;{year_max}
  &nbsp;&middot;&nbsp;
  <a href="https://frprado.github.io" target="_blank"
     style="color:{C_PRIMARY}; text-decoration: none; font-weight: 600;">
    Frederico Prado
  </a>
</div>

</body>
</html>"""

out_path = os.path.join(_DIR, "report.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nReport saved → {out_path}")

# ── PDF / PNG export via Playwright ───────────────────────────────────────────
if _args.pdf or _args.png:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("\nPlaywright not installed. Run:\n  pip install playwright && playwright install chromium")
    else:
        file_url = f"file://{os.path.abspath(out_path)}"
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(file_url)
            page.wait_for_load_state("load")
            # Give Plotly charts time to finish rendering after JS executes
            page.wait_for_timeout(4000)

            if _args.pdf:
                pdf_path = os.path.join(_DIR, "report.pdf")
                page.pdf(
                    path=pdf_path,
                    format="A3",
                    landscape=True,
                    print_background=True,
                    margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
                )
                print(f"PDF saved   → {pdf_path}")

            if _args.png:
                png_path = os.path.join(_DIR, "report.png")
                page.screenshot(path=png_path, full_page=True)
                print(f"PNG saved   → {png_path}")

            browser.close()

webbrowser.open(f"file://{os.path.abspath(out_path)}")
