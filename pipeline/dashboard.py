# dashboard.py
"""
Tabbed dashboard for linguistics publications (OpenAlex data).
Tabs:
 - Overview: publications trend + topic distribution + world map
 - Journals & Countries: bar + 100%-normalised trend (Top 5)
 - Languages & Concepts: bar + 100%-normalised trend (Top 5)
 - Papers: filterable table

Trend charts use a 100% normalised stacked area so the reader sees
*share* change over time (e.g. "did English go from 80% → 30%?"),
not just raw paper counts which conflate growth with composition shift.
"""
import ast
import os
import numpy as np
import pandas as pd
import pycountry

import plotly.express as px

from dash import Dash, html, dcc, Input, Output, State, no_update, callback_context, dash_table
import dash_bootstrap_components as dbc

# ─────────────────────────────────────────
# Palette — modern dark-sidebar SaaS
# ─────────────────────────────────────────
C_BG      = "#F1F5F9"   # slate-100 — page background
C_CARD    = "#FFFFFF"   # white — chart / card surface
C_SIDEBAR = "#0F172A"   # slate-900 — sidebar
C_PRIMARY = "#6366F1"   # indigo-500 — primary accent
C_TEXT    = "#1E293B"   # slate-800 — body text
C_MUTED   = "#64748B"   # slate-500 — secondary / labels
C_BORDER  = "#E2E8F0"   # slate-200 — dividers

PALETTE = [
    "#6366F1",  # indigo
    "#10B981",  # emerald
    "#F59E0B",  # amber
    "#EF4444",  # rose
    "#3B82F6",  # blue
    "#8B5CF6",  # violet
    "#06B6D4",  # cyan
    "#F97316",  # orange
]

# keep old aliases used downstream
C_TEAL   = C_SIDEBAR
C_NAVY   = C_TEXT
C_ORANGE = "#F59E0B"
CHART_BG = C_CARD
TOP_N    = 5


def style_fig(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=C_CARD,
        plot_bgcolor=C_CARD,
        margin=dict(l=8, r=8, t=44, b=8),
        font=dict(family="'Inter', system-ui, sans-serif", size=12, color=C_TEXT),
        title_font=dict(size=13, color=C_TEXT, family="'Inter', system-ui, sans-serif"),
        colorway=PALETTE,
        showlegend=False,
    )
    return fig


def style_fig_with_legend(fig):
    style_fig(fig)
    fig.update_layout(
        showlegend=True,
        margin=dict(l=8, r=150, t=44, b=8),
        legend=dict(
            orientation="v",
            yanchor="middle", y=0.5,
            xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=C_BORDER,
            borderwidth=1,
            font=dict(size=10, color=C_TEXT),
        ),
    )
    return fig


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def safe_literal_eval(x):
    if isinstance(x, (list, dict)):
        return x
    if not isinstance(x, str):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return x


def parse_concepts_field(x):
    v = safe_literal_eval(x)
    if isinstance(v, list):
        return [str(e).strip() for e in v if e is not None and str(e).strip()]
    if isinstance(v, str):
        return [p.strip() for p in v.split(",") if p.strip()]
    return []


def normalize_languages_field(x):
    if pd.isna(x):
        return []
    parts = [p.strip() for p in str(x).split(",")]
    parts = [p for p in parts if p and p.lower() != "unknown"]
    normalized = [p.lower().title() for p in parts]
    seen, out = set(), []
    for p in normalized:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def iso2_to_alpha3(code):
    if not isinstance(code, str) or not code.strip():
        return None
    c = code.strip().upper()
    for lookup in ("alpha_2", "alpha_3"):
        try:
            country = pycountry.countries.get(**{lookup: c})
            if country and hasattr(country, "alpha_3"):
                return country.alpha_3
        except Exception:
            pass
    return None


def iso2_to_name(code):
    if not isinstance(code, str) or not code.strip():
        return None
    c = code.strip().upper()
    for lookup in ("alpha_2", "alpha_3"):
        try:
            country = pycountry.countries.get(**{lookup: c})
            if country:
                return country.name
        except Exception:
            pass
    return code


# ─────────────────────────────────────────
# Load & prepare data
# ─────────────────────────────────────────
TOP_JOURNALS_LIST = [
    "Language",
    "Linguistic Inquiry",
    "Lingua",
    "Journal of Linguistics",
    "Natural language and linguistic theory"
]

KEEP_TOPICS = {
    "Syntax, Semantics, Linguistic Variation",
    "Linguistic research and analysis",
    "Linguistics and language evolution",
    "Historical Linguistics and Language Studies",
    "Linguistics and Language Analysis",
    "Linguistics and Language Studies",
}

_data_path = os.path.join(os.path.dirname(__file__), "data.parquet")
if not os.path.exists(_data_path):
    _data_path = os.path.join(os.path.dirname(__file__), "data.csv")
df = pd.read_parquet(_data_path) if _data_path.endswith(".parquet") else pd.read_csv(_data_path)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df = df[df["primary_topic"].isin(KEEP_TOPICS)].copy()

# Drop rows whose journal is confirmed non-linguistics (if allowlist exists)
_allowlist_path = os.path.join(os.path.dirname(__file__), "journal_allowlist.csv")
if os.path.exists(_allowlist_path):
    _al = pd.read_csv(_allowlist_path)
    # support both old 'linguistics' column and new 'qualified' column
    _qual_col = "qualified" if "qualified" in _al.columns else "linguistics"
    _good = set(_al.loc[_al[_qual_col] == True, "journal"].dropna())
    # Keep rows with no journal assigned, or whose journal is explicitly qualified.
    # Any row with a journal name NOT in the allowlist is dropped.
    # Keep only rows with a journal that is explicitly qualified
    _has_journal = df["journal"].notna() & (df["journal"].str.strip() != "")
    df = df[_has_journal & df["journal"].isin(_good)].copy()

expected_cols = {"title", "year", "doi", "cited_by", "journal", "concepts",
                 "top_concept", "paper_country", "abstract", "primary_topic"}
missing = expected_cols - set(df.columns)
if missing:
    print(f"Warning: missing columns (will continue): {missing}")

df["year"] = pd.to_numeric(df.get("year", pd.Series()), errors="coerce")
df = df.dropna(subset=["year"]).copy()
df["year"] = df["year"].astype(int)

df["cited_by"] = (
    pd.to_numeric(df.get("cited_by", pd.Series([0] * len(df))), errors="coerce")
    .fillna(0).astype(int)
)
df["paper_country"]      = df.get("paper_country",      pd.Series([""] * len(df))).fillna("")
df["language_mentioned"] = df.get("language_mentioned", pd.Series([""] * len(df))).fillna("")
df["journal"]            = df.get("journal",            pd.Series([""] * len(df))).fillna("")
df["title"]              = df.get("title",              pd.Series([""] * len(df))).fillna("")
df["primary_topic"]      = df.get("primary_topic",      pd.Series([""] * len(df))).fillna("")

df["concept_list"]   = df.get("concepts", pd.Series([[]] * len(df))).apply(parse_concepts_field)
df["language_list"]  = df["language_mentioned"].apply(normalize_languages_field)
df["country_name"]   = df["paper_country"].apply(lambda x: iso2_to_name(x) if isinstance(x, str) else None)
df["country_alpha3"] = df["paper_country"].apply(lambda x: iso2_to_alpha3(x) if isinstance(x, str) else None)

df_lang     = df.explode("language_list").copy()
df_lang["language_list"] = df_lang["language_list"].fillna("")

# Load language family lookup (if linglist has a 'family' column)
_linglist_path = os.path.join(os.path.dirname(__file__), "linglist.csv")
_linglist = pd.read_csv(_linglist_path)
if "family" in _linglist.columns:
    _lang_family = (
        _linglist[["language", "family"]]
        .dropna()
        .assign(language=lambda d: d["language"].str.strip().str.lower().str.title())
        .drop_duplicates(subset=["language"])
        .set_index("language")["family"]
        .to_dict()
    )
    df_lang["family"] = df_lang["language_list"].str.strip().str.lower().str.title().map(_lang_family).fillna("Other/Unclassified")
else:
    df_lang["family"] = "Unknown"

df_concepts = df.explode("concept_list").copy()
df_concepts["concept_list"] = df_concepts["concept_list"].fillna("")

# ─────────────────────────────────────────
# Dropdown options
# ─────────────────────────────────────────
def _opts(series):
    return [{"label": v, "value": v}
            for v in sorted(series.dropna().unique()) if str(v).strip()]

country_options  = _opts(df["country_name"])
journal_options  = _opts(df["journal"])
language_options = _opts(df_lang["language_list"])
concept_options  = _opts(df_concepts["concept_list"])
_topic_col    = "subfield" if "subfield" in df.columns else "primary_topic"
topic_options = _opts(df[_topic_col])

# ─────────────────────────────────────────
# App & layout
# ─────────────────────────────────────────
def kpi_card(label, component_id, accent=C_PRIMARY, width=3, value_size="1.5rem"):
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.P(label, style={
                    "fontSize": "9px", "fontWeight": "700",
                    "textTransform": "uppercase", "letterSpacing": "0.1em",
                    "color": C_MUTED, "marginBottom": "8px", "lineHeight": "1.4",
                }),
                html.Div("—", id=component_id, style={
                    "fontWeight": "700", "color": C_TEXT,
                    "fontSize": value_size, "lineHeight": "1.2",
                    "wordBreak": "break-word",
                }),
            ], style={"padding": "14px 18px"}),
            style={
                "backgroundColor": C_CARD,
                "borderRadius": "10px",
                "border": "none",
                "borderLeft": f"4px solid {accent}",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04)",
                "height": "100%",
            },
        ),
        width=width,
    )


app = Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
])
server = app.server

GH = "380px"

overview_tab = dbc.Tab(label="Overview", tab_id="tab_overview", children=[
    dbc.Row([
        kpi_card("Total publications",              "kpi_total",    C_PRIMARY,  width=2, value_size="1.8rem"),
        kpi_card("Country with the most publications",  "kpi_country",  "#10B981",  width=3, value_size="1.3rem"),
        kpi_card("Most studied language",           "kpi_language", "#F59E0B",  width=3, value_size="1.3rem"),
        kpi_card("Subfield with the most publications",                    "kpi_subfield", "#EF4444",  width=4, value_size="1.1rem"),
    ], class_name="mt-3 g-3"),
    dbc.Row(dbc.Col(dcc.Graph(id="overview_trend", style={"height": GH})), class_name="mt-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="topic_bar",   style={"height": GH}), width=5),
        dbc.Col(dcc.Graph(id="topic_trend", style={"height": GH}), width=7),
    ], class_name="mt-2"),
    dbc.Row(dbc.Col(dcc.Graph(id="overview_map", style={"height": "480px"})), class_name="mt-2"),
])

journals_countries_tab = dbc.Tab(label="Journals & Countries", tab_id="tab_journals_countries", children=[
    dbc.Row([
        dbc.Col(dcc.Graph(id="journal_bar",   style={"height": GH}), width=6),
        dbc.Col(dcc.Graph(id="country_bar",   style={"height": GH}), width=6),
    ], class_name="mt-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="journal_trend", style={"height": GH}), width=6),
        dbc.Col(dcc.Graph(id="country_trend", style={"height": GH}), width=6),
    ], class_name="mt-2"),
])

langs_concepts_tab = dbc.Tab(label="Languages", tab_id="tab_langs_concepts", children=[
    dbc.Row(
        dbc.Col(dcc.Graph(id="lang_family_treemap", style={"height": "500px"})),
        class_name="mt-3",
    ),
    dbc.Row(
        dbc.Col(dcc.Graph(id="lang_trend", style={"height": GH})),
        class_name="mt-2",
    ),
    dbc.Row(
        dbc.Col(dcc.Graph(id="diversity_trend", style={"height": "280px"})),
        class_name="mt-2",
    ),
])

top_journals_tab = dbc.Tab(label="Top Journals", tab_id="tab_top_journals", children=[
    dbc.Row([
        kpi_card("Most linguistically diverse journal", "kpi_j_div_top", C_PRIMARY,  width=5, value_size="1.1rem"),
        kpi_card("Least linguistically diverse journal", "kpi_j_div_bot", C_MUTED,   width=5, value_size="1.1rem"),
    ], class_name="mt-3 g-3"),
    dbc.Row(
        dbc.Col(dcc.Graph(id="top_journals_ts",       style={"height": GH})),
        class_name="mt-3",
    ),
    dbc.Row([
        dbc.Col(dcc.Graph(id="top_journals_diversity",  style={"height": GH}), width=5),
        dbc.Col(dcc.Graph(id="top_journals_lang_dist",  style={"height": GH}), width=7),
    ], class_name="mt-2"),
])

papers_tab = dbc.Tab(label="Papers", tab_id="tab_papers", children=[
    dbc.Row(dbc.Col(
        dash_table.DataTable(
            id="papers_table",
            columns=[], data=[],
            page_size=20,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left", "padding": "8px 12px",
                "maxWidth": "420px", "overflow": "hidden",
                "textOverflow": "ellipsis",
                "fontFamily": "'Inter', system-ui, sans-serif", "fontSize": "13px",
                "color": C_TEXT, "border": f"1px solid {C_BORDER}",
            },
            style_header={
                "fontWeight": "600",
                "backgroundColor": C_SIDEBAR,
                "color": "#F1F5F9",
                "border": f"1px solid {C_SIDEBAR}",
                "fontFamily": "'Inter', system-ui, sans-serif", "fontSize": "11px",
                "textTransform": "uppercase", "letterSpacing": "0.06em",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": C_BG},
            ],
        )
    ), class_name="mt-3"),
])

LABEL_STYLE = {
    "fontWeight": "600", "fontSize": "10px",
    "color": "#94A3B8", "marginBottom": "5px",
    "textTransform": "uppercase", "letterSpacing": "0.08em",
    "display": "block",
}

_dd_style = {
    "fontSize": "12px",
    "borderRadius": "6px",
    "border": f"1px solid #334155",
    "backgroundColor": "#1E293B",
    "color": "#F1F5F9",
}

controls = html.Div([
    html.Div([
        html.Div("Filters", style={
            "fontSize": "12px", "fontWeight": "700",
            "color": "#94A3B8", "textTransform": "uppercase",
            "letterSpacing": "0.1em", "marginBottom": "20px",
        }),
        html.Label("Year range", style=LABEL_STYLE),
        dcc.RangeSlider(
            id="year_range",
            min=int(df["year"].min()), max=int(df["year"].max()), step=1,
            value=[int(df["year"].min()), int(df["year"].max())],
            marks={int(y): {"label": str(int(y)), "style": {"fontSize": "9px", "color": "#64748B"}}
                   for y in sorted(df["year"].unique()) if int(y) % 10 == 0},
            tooltip={"placement": "bottom"},
        ),
        html.Div(className="mb-3"),
        html.Label("Topic", style=LABEL_STYLE),
        dcc.Dropdown(id="topic_filter", options=topic_options, multi=True,
                     placeholder="All topics", className="mb-3",
                     style=_dd_style),
        html.Label("Country", style=LABEL_STYLE),
        dcc.Dropdown(id="country_filter", options=country_options, multi=True,
                     placeholder="All countries", className="mb-3",
                     style=_dd_style),
        html.Label("Journal", style=LABEL_STYLE),
        dcc.Dropdown(id="journal_filter", options=journal_options, multi=True,
                     placeholder="All journals", className="mb-3",
                     style=_dd_style),
        html.Label("Language studied", style=LABEL_STYLE),
        dcc.Dropdown(id="language_filter", options=language_options, multi=True,
                     placeholder="All languages", className="mb-3",
                     style=_dd_style),
        html.Label("Concept", style=LABEL_STYLE),
        dcc.Dropdown(id="concept_filter", options=concept_options, multi=True,
                     placeholder="All concepts", className="mb-3",
                     style=_dd_style),
        html.Label("Title search", style=LABEL_STYLE),
        dcc.Input(id="title_search", type="text", placeholder="Search titles…",
                  style={
                      "width": "100%", "padding": "7px 10px", "fontSize": "12px",
                      "borderRadius": "6px", "border": "1px solid #334155",
                      "backgroundColor": "#1E293B", "color": "#F1F5F9",
                      "marginBottom": "20px", "boxSizing": "border-box",
                  }),
        dbc.Button("Reset filters", id="reset_btn", size="sm", className="w-100",
                   style={
                       "backgroundColor": C_PRIMARY, "border": "none", "color": "white",
                       "fontWeight": "600", "borderRadius": "6px", "fontSize": "12px",
                       "padding": "8px",
                   }),
        html.Div(id="selected_info", className="mt-2",
                 style={"fontSize": "11px", "color": "#64748B"}),
    ], style={"padding": "24px 16px"}),
], style={
    "backgroundColor": C_SIDEBAR,
    "borderRadius": "12px",
    "minHeight": "100%",
    "position": "sticky",
    "top": "0",
})

app.layout = dbc.Container([
    dbc.Row(dbc.Col(
        html.Div([
            html.Div([
                html.H4("Linguistics Research", className="mb-0 fw-bold",
                        style={"color": C_TEXT, "letterSpacing": "-0.02em"}),
                html.Span(" Dashboard", style={"color": C_PRIMARY, "fontWeight": "700",
                                               "fontSize": "1.3rem"}),
            ], style={"display": "inline-flex", "alignItems": "baseline", "gap": "2px"}),
            html.P("Global linguistics publications via OpenAlex · 1960–2024",
                   className="mb-0 mt-1", style={"fontSize": "12px", "color": C_MUTED}),
        ], style={"padding": "20px 4px 16px"}),
    ), style={"borderBottom": f"1px solid {C_BORDER}", "marginBottom": "20px"}),
    dbc.Row([
        dbc.Col(controls, width=3),
        dbc.Col(
            dbc.Tabs(
                [overview_tab, journals_countries_tab, langs_concepts_tab, top_journals_tab, papers_tab],
                id="tabs", active_tab="tab_overview",
            ),
            width=9,
        ),
    ], className="g-3"),
    dcc.Store(id="selection_store",
              data={"language": None, "journal": None, "country": None,
                    "concept": None, "topic": None}),
], fluid=True, style={
    "minHeight": "100vh",
    "backgroundColor": C_BG,
    "fontFamily": "'Inter', system-ui, sans-serif",
    "padding": "0 24px 40px",
})


# ─────────────────────────────────────────
# Filtering helper
# ─────────────────────────────────────────
def apply_filters(df_main, df_lang_exploded, df_concepts_exploded,
                  year_range, topics, countries, journals, languages, concepts,
                  title_search, selection_store):
    d = df_main.copy()
    if year_range is None:
        year_range = [int(df["year"].min()), int(df["year"].max())]
    d = d[(d["year"] >= year_range[0]) & (d["year"] <= year_range[1])]

    if topics:
        _tc = "subfield" if "subfield" in d.columns else "primary_topic"
        d = d[d[_tc].isin(topics)]
    if countries:
        d = d[d["country_name"].isin(countries)]
    if journals:
        d = d[d["journal"].isin(journals)]
    if title_search:
        p = str(title_search).lower()
        d = d[d["title"].fillna("").str.lower().str.contains(p)]
    if languages:
        idx = df_lang_exploded[df_lang_exploded["language_list"].isin(languages)].index.unique()
        d = d.loc[d.index.isin(idx)]
    if concepts:
        idx = df_concepts_exploded[df_concepts_exploded["concept_list"].isin(concepts)].index.unique()
        d = d.loc[d.index.isin(idx)]

    sel = selection_store or {}
    if sel.get("topic"):
        _tc = "subfield" if "subfield" in d.columns else "primary_topic"
        d = d[d[_tc] == sel["topic"]]
    if sel.get("language"):
        idx = df_lang_exploded[df_lang_exploded["language_list"] == sel["language"]].index.unique()
        d = d.loc[d.index.isin(idx)]
    if sel.get("journal"):
        d = d[d["journal"] == sel["journal"]]
    if sel.get("country"):
        d = d[d["country_name"] == sel["country"]]
    if sel.get("concept"):
        idx = df_concepts_exploded[df_concepts_exploded["concept_list"] == sel["concept"]].index.unique()
        d = d.loc[d.index.isin(idx)]

    return d


# ─────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────
def make_bar(counts_df, x_col, y_col, title):
    """Horizontal bar — each bar coloured with a cycling PALETTE colour."""
    n = len(counts_df)
    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]
    fig = px.bar(counts_df, x=x_col, y=y_col, orientation="h", title=title)
    fig.update_traces(marker_color=colors)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="total papers")
    return style_fig(fig)


def make_norm_trend(trend_df, x_col, y_col, color_col, title, category_order=None):
    """100% normalised stacked area chart.

    Shows the *share* of each category per year (all traces sum to 100 % at
    every x tick), so the reader can see composition shifts over time — e.g.
    English going from 70 % → 30 % — independently of total paper volume.
    """
    if trend_df.empty:
        return style_fig(px.scatter(title=title))
    if category_order is None:
        category_order = (trend_df.groupby(color_col)["count"].sum()
                          .sort_values(ascending=False).index.tolist())
    fig = px.area(
        trend_df, x=x_col, y=y_col, color=color_col,
        title=title,
        color_discrete_sequence=PALETTE,
        category_orders={color_col: category_order},
        line_shape="spline",
    )
    # groupnorm="percent" makes every year sum to 100 %
    fig.update_traces(stackgroup="one", groupnorm="percent", line=dict(width=0.6))
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="% share", range=[0, 100],
                     ticksuffix="%", showgrid=True,
                     gridcolor=C_BORDER, gridwidth=1)
    style_fig_with_legend(fig)
    return fig


def bar_and_trend(series, label, series_name, dff_ref, top_n=20, trend_n=TOP_N):
    s = series.dropna()
    s = s[s.astype(str).str.strip() != ""]
    counts = s.value_counts().reset_index()
    counts.columns = [label, "count"]
    top_cats = counts[label].head(trend_n).tolist()
    bar_fig = make_bar(counts.head(top_n), "count", label, f"Top {label}s")
    if top_cats:
        trend_df = (
            dff_ref[dff_ref[series_name].isin(top_cats)]
            .groupby(["year", series_name]).size().reset_index(name="count")
        )
        trend_fig = make_norm_trend(trend_df, "year", "count", series_name,
                                    f"{label} share over time · top {trend_n}",
                                    category_order=top_cats)
    else:
        trend_fig = style_fig(px.scatter(title=f"{label} share over time · top {trend_n}"))
    return bar_fig, trend_fig


def compute_diversity(lang_exploded):
    """Shannon entropy (bits) of language distribution per year."""
    rows = lang_exploded[lang_exploded["language_list"].str.strip() != ""]
    if rows.empty:
        return pd.DataFrame(columns=["year", "entropy"])
    def _h(g):
        c = g.value_counts()
        p = c / c.sum()
        return float(-(p * np.log2(p)).sum())
    return rows.groupby("year")["language_list"].apply(_h).reset_index(name="entropy")


def exploded_bar_and_trend(exploded_series, exploded_df, year_col, color_col,
                            label, top_n=20, trend_n=TOP_N):
    s = exploded_series.dropna()
    s = s[s.astype(str).str.strip() != ""]
    counts = s.value_counts().reset_index()
    counts.columns = [label, "count"]
    top_cats = counts[label].head(trend_n).tolist()
    bar_fig = make_bar(counts.head(top_n), "count", label, f"Top {label}s")
    if top_cats:
        subset = exploded_df[exploded_df[color_col].isin(top_cats)]
        trend_df = subset.groupby([year_col, color_col]).size().reset_index(name="count")
        trend_fig = make_norm_trend(trend_df, year_col, "count", color_col,
                                    f"{label} share over time · top {trend_n}",
                                    category_order=top_cats)
    else:
        trend_fig = style_fig(px.scatter(title=f"{label} share over time · top {trend_n}"))
    return bar_fig, trend_fig


# ─────────────────────────────────────────
# Main callback
# ─────────────────────────────────────────
@app.callback(
    Output("kpi_total",       "children"),
    Output("kpi_country",     "children"),
    Output("kpi_language",    "children"),
    Output("kpi_subfield",    "children"),

    Output("overview_trend",  "figure"),
    Output("topic_bar",       "figure"),
    Output("topic_trend",     "figure"),
    Output("overview_map",    "figure"),

    Output("journal_bar",     "figure"),
    Output("journal_trend",   "figure"),
    Output("country_bar",     "figure"),
    Output("country_trend",   "figure"),

    Output("lang_trend",          "figure"),
    Output("lang_family_treemap", "figure"),
    Output("diversity_trend",     "figure"),

    Output("papers_table",    "columns"),
    Output("papers_table",    "data"),

    Input("year_range",       "value"),
    Input("topic_filter",     "value"),
    Input("country_filter",   "value"),
    Input("journal_filter",   "value"),
    Input("language_filter",  "value"),
    Input("concept_filter",   "value"),
    Input("title_search",     "value"),
    Input("selection_store",  "data"),
)
def update_all(year_range, topics, countries, journals, languages, concepts,
               title_search, selection_store):
    dff = apply_filters(df, df_lang, df_concepts,
                        year_range, topics, countries, journals, languages, concepts,
                        title_search, selection_store)

    # ── KPI cards ──
    kpi_total = f"{len(dff):,}"
    _country_counts = dff["country_name"].dropna()
    _country_counts = _country_counts[_country_counts.str.strip() != ""].value_counts()
    kpi_country = _country_counts.index[0] if not _country_counts.empty else "—"

    _lang_series_kpi = df_lang.loc[dff.index]["language_list"].dropna()
    _lang_series_kpi = _lang_series_kpi[_lang_series_kpi.str.strip() != ""].value_counts()
    kpi_language = _lang_series_kpi.index[0] if not _lang_series_kpi.empty else "—"

    _tc = "subfield" if "subfield" in dff.columns else "primary_topic"
    _subfield_counts = dff[_tc].dropna().value_counts()
    kpi_subfield = _subfield_counts.index[0] if not _subfield_counts.empty else "—"

    # ── overview trend (absolute — provides growth context) ──
    pubs_by_year = dff.groupby("year").size().reset_index(name="count")
    fig_overview = px.area(
        pubs_by_year, x="year", y="count",
        title="Papers per year",
        color_discrete_sequence=[C_PRIMARY],
        line_shape="spline",
    )
    fig_overview.update_traces(line=dict(width=1.5, color=C_PRIMARY),
                               fillcolor="rgba(99,102,241,0.12)")
    fig_overview.update_xaxes(title_text="")
    fig_overview.update_yaxes(title_text="papers",
                               gridcolor=C_BORDER, gridwidth=1)
    style_fig(fig_overview)

    # ── topic distribution ──
    _topic_col = "subfield" if "subfield" in dff.columns else "primary_topic"
    topic_bar_fig, topic_trend_fig = bar_and_trend(
        dff[_topic_col], "subfield", _topic_col, dff, top_n=20)

    # ── world map ──
    map_df = (dff[~dff["country_alpha3"].isna()]
              .groupby("country_alpha3").size().reset_index(name="count"))
    if not map_df.empty:
        fig_map = px.choropleth(
            map_df, locations="country_alpha3", color="count",
            color_continuous_scale=[[0, "#E0E7FF"], [0.4, "#6366F1"], [1, "#1E293B"]],
            projection="natural earth",
            labels={"count": "Papers"}, title="Papers per country",
        )
        fig_map.update_geos(showframe=False, showcoastlines=True,
                            coastlinecolor=C_BORDER, landcolor="#F8FAFC",
                            oceancolor="#EFF6FF", showocean=True)
    else:
        fig_map = px.choropleth(locations=[], color=[], title="Papers per country")
    style_fig(fig_map)
    fig_map.update_layout(margin=dict(l=0, r=0, t=44, b=0))

    # ── journals & countries ──
    journal_bar_fig, journal_trend_fig = bar_and_trend(
        dff["journal"], "journal", "journal", dff)
    country_bar_fig, country_trend_fig = bar_and_trend(
        dff["country_name"], "country", "country_name", dff)

    # ── languages ──
    lang_series   = df_lang.loc[dff.index]["language_list"]
    lang_exploded = df_lang.loc[dff.index].copy()
    _, lang_trend_fig = exploded_bar_and_trend(
        lang_series, lang_exploded, "year", "language_list", "language")

    # ── language family treemap ──
    _EXCLUDE_FAMILIES = {"Other/Unclassified", "Uncoded", "Undetermined"}
    _fam = (
        lang_exploded[
            lang_exploded["language_list"].str.strip().astype(bool) &
            ~lang_exploded["family"].isin(_EXCLUDE_FAMILIES)
        ]
        .groupby(["family", "language_list"])
        .size()
        .reset_index(name="count")
        .rename(columns={"language_list": "language"})
    )
    if not _fam.empty:
        lang_family_fig = px.treemap(
            _fam, path=["family", "language"], values="count",
            title="Languages studied · by family",
            color_discrete_sequence=PALETTE,
        )
        lang_family_fig.update_traces(
            textinfo="label+percent parent",
            hovertemplate="<b>%{label}</b><br>%{value} papers<extra></extra>",
        )
        style_fig(lang_family_fig)
        lang_family_fig.update_layout(showlegend=False, margin=dict(l=8, r=8, t=44, b=8))
    else:
        lang_family_fig = style_fig(px.scatter(title="Languages studied · by family"))

    # ── language diversity trend ──
    div_df = compute_diversity(df_lang.loc[dff.index])
    if not div_df.empty:
        fig_diversity = px.line(
            div_df, x="year", y="entropy",
            title="Language diversity index · per year",
            color_discrete_sequence=[C_PRIMARY],
            line_shape="spline",
        )
        fig_diversity.update_traces(line=dict(width=2, color=C_PRIMARY))
        fig_diversity.update_xaxes(title_text="")
        fig_diversity.update_yaxes(title_text="Shannon H (bits)")
        style_fig(fig_diversity)
    else:
        fig_diversity = style_fig(px.scatter(title="Language diversity index · per year"))

    # ── papers table ──
    table_cols = ["year", "title", "journal", "cited_by", "paper_country", "doi"]
    available  = [c for c in table_cols if c in dff.columns]
    table_df   = dff[available].copy()
    columns    = [{"name": c, "id": c} for c in table_df.columns]
    data       = table_df.to_dict("records")

    return (
        kpi_total, kpi_country, kpi_language, kpi_subfield,
        fig_overview,
        topic_bar_fig, topic_trend_fig,
        fig_map,
        journal_bar_fig, journal_trend_fig,
        country_bar_fig, country_trend_fig,
        lang_trend_fig, lang_family_fig, fig_diversity,
        columns, data,
    )


# ─────────────────────────────────────────
# Click-to-filter + reset callback
# ─────────────────────────────────────────
@app.callback(
    Output("selection_store",  "data"),
    Output("year_range",       "value"),
    Output("topic_filter",     "value"),
    Output("country_filter",   "value"),
    Output("journal_filter",   "value"),
    Output("language_filter",  "value"),
    Output("concept_filter",   "value"),
    Output("title_search",     "value"),

    Input("topic_bar",         "clickData"),
    Input("journal_bar",       "clickData"),
    Input("country_bar",       "clickData"),
    Input("reset_btn",         "n_clicks"),
    State("selection_store",   "data"),
    prevent_initial_call=True,
)
def update_filters(topic_click, journal_click, country_click,
                   reset_click, store):
    ctx = callback_context
    if not ctx.triggered:
        return (no_update,) * 8
    trig  = ctx.triggered[0]["prop_id"].split(".")[0]
    store = store or {"language": None, "journal": None, "country": None,
                      "concept": None, "topic": None}
    _nu = (no_update,) * 7

    if trig == "reset_btn":
        return (
            {"language": None, "journal": None, "country": None,
             "concept": None, "topic": None},
            [int(df["year"].min()), int(df["year"].max())],
            None, None, None, None, None, "",
        )

    def _label(cd):
        if not cd:
            return None
        pt = cd["points"][0]
        return pt.get("y") or pt.get("label") or pt.get("x")

    clicks = {
        "topic_bar":   ("topic",   topic_click),
        "journal_bar": ("journal", journal_click),
        "country_bar": ("country", country_click),
    }
    if trig in clicks:
        key, cd = clicks[trig]
        label = _label(cd)
        if label:
            new = {"language": None, "journal": None, "country": None,
                   "concept": None, "topic": None}
            new[key] = label
            return (new, *_nu)

    return (store, *_nu)


# ─────────────────────────────────────────
# Top Journals callback
# ─────────────────────────────────────────
@app.callback(
    Output("kpi_j_div_top",         "children"),
    Output("kpi_j_div_bot",         "children"),
    Output("top_journals_ts",        "figure"),
    Output("top_journals_diversity", "figure"),
    Output("top_journals_lang_dist", "figure"),
    Input("year_range", "value"),
)
def update_top_journals(year_range):
    if year_range is None:
        year_range = [int(df["year"].min()), int(df["year"].max())]

    dff_j = df[
        df["journal"].isin(TOP_JOURNALS_LIST) &
        (df["year"] >= year_range[0]) &
        (df["year"] <= year_range[1])
    ].copy()
    dfl_j = df_lang[
        df_lang["journal"].isin(TOP_JOURNALS_LIST) &
        (df_lang["year"] >= year_range[0]) &
        (df_lang["year"] <= year_range[1])
    ].copy()

    # Publications per year per journal
    ts_df = dff_j.groupby(["year", "journal"]).size().reset_index(name="count")
    if not ts_df.empty:
        fig_ts = px.line(
            ts_df, x="year", y="count", color="journal",
            title="Publications per year · curated journals",
            color_discrete_sequence=PALETTE,
            line_shape="spline",
        )
        fig_ts.update_traces(line=dict(width=2))
        fig_ts.update_xaxes(title_text="")
        fig_ts.update_yaxes(title_text="papers")
        style_fig_with_legend(fig_ts)
    else:
        fig_ts = style_fig(px.scatter(title="Publications per year · curated journals"))

    # Diversity index per journal
    def _h(g):
        g = g[g.str.strip() != ""]
        if g.empty:
            return 0.0
        c = g.value_counts()
        p = c / c.sum()
        return float(-(p * np.log2(p)).sum())

    non_empty = dfl_j[dfl_j["language_list"].str.strip() != ""]
    if not non_empty.empty:
        div_j = (
            non_empty.groupby("journal")["language_list"]
            .apply(_h)
            .reset_index(name="H")
            .sort_values("H", ascending=False)
        )
        kpi_top = f"{div_j.iloc[0]['journal']} ({div_j.iloc[0]['H']:.2f} bits)"
        kpi_bot = f"{div_j.iloc[-1]['journal']} ({div_j.iloc[-1]['H']:.2f} bits)" if len(div_j) > 1 else "—"

        colors = [PALETTE[i % len(PALETTE)] for i in range(len(div_j))]
        fig_div = px.bar(
            div_j, x="H", y="journal", orientation="h",
            title="Language diversity index per journal",
        )
        fig_div.update_traces(marker_color=colors)
        fig_div.update_yaxes(autorange="reversed", title_text="")
        fig_div.update_xaxes(title_text="Shannon H (bits)")
        style_fig(fig_div)
    else:
        kpi_top = kpi_bot = "—"
        fig_div = style_fig(px.scatter(title="Language diversity index per journal"))

    # Language distribution per journal (top 8 languages stacked)
    top_langs_overall = (
        non_empty["language_list"].value_counts().head(8).index.tolist()
        if not non_empty.empty else []
    )
    if top_langs_overall:
        lang_dist_df = (
            non_empty[non_empty["language_list"].isin(top_langs_overall)]
            .groupby(["journal", "language_list"]).size().reset_index(name="count")
        )
        fig_lang = px.bar(
            lang_dist_df, x="count", y="journal", color="language_list",
            orientation="h",
            title="Language distribution per journal (top 8 languages)",
            color_discrete_sequence=PALETTE,
            barmode="stack",
        )
        fig_lang.update_yaxes(title_text="")
        fig_lang.update_xaxes(title_text="papers")
        style_fig_with_legend(fig_lang)
    else:
        fig_lang = style_fig(px.scatter(title="Language distribution per journal"))

    return kpi_top, kpi_bot, fig_ts, fig_div, fig_lang


# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Starting dashboard on http://127.0.0.1:8050")
    app.run(debug=False)
