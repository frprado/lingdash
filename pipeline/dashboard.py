# dashboard_tabbed_final.py
"""
Final tabbed dashboard:
Tabs:
 - Overview: publications trend + world choropleth (publications per country)
 - Journals & Countries: bar (horizontal) + trend (Top 8) each
 - Languages, Concepts & Authors: bar + trend (Top 8) each; authors also show citations per author

Notes:
 - Assumes 'paper_country' uses ISO-2 codes (US, BR, etc.)
 - 'authors' column contains Python-dict-like strings like "{1: 'A', 2: 'B'}" or actual dicts
 - 'concepts' may be Python-list-like strings (["a","b"]) — parsed with ast.literal_eval safely
 - Unknown/blank entries are excluded from visuals & dropdowns, but remain in the raw table
"""
import ast
import math
import pandas as pd
import numpy as np
import pycountry

import plotly.express as px

from dash import Dash, html, dcc, Input, Output, State, no_update, callback_context
import dash_bootstrap_components as dbc
import dash_table

# -------------------------
# Config
# -------------------------
CSV_PATH = r"C:\Users\fblprado\Documents\GitHub\lingdash\pipeline\data.csv"  # set to your file
TOP_N = 8  # Top N for trend line charts

# -------------------------
# Helpers: parsing & country conversions
# -------------------------
def safe_literal_eval(x):
    """Try to parse Python literals (lists, dicts). Return original if fail."""
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
        return [str(e).strip() for e in v if e is not None and str(e).strip() != ""]
    if isinstance(v, str):
        # fallback: comma-separated?
        parts = [p.strip() for p in v.split(",") if p.strip()]
        return parts
    return []

def parse_authors_field(x):
    """If x is a dict-like string, return list of values (unordered)."""
    v = safe_literal_eval(x)
    if isinstance(v, dict):
        # take values, ignore keys; preserve insertion order if dict
        return [str(name).strip() for name in v.values() if name is not None and str(name).strip() != ""]
    if isinstance(v, list):
        return [str(e).strip() for e in v if e is not None and str(e).strip() != ""]
    if isinstance(v, str):
        # try to parse: maybe semicolon/comma-separated
        parts = [p.strip() for p in v.replace(";", ",").split(",") if p.strip()]
        return parts
    return []

def normalize_languages_field(x):
    if pd.isna(x):
        return []
    parts = [p.strip() for p in str(x).split(",")]
    parts = [p for p in parts if p and p.lower() != "unknown"]
    normalized = [p.lower().title() for p in parts]
    seen = set()
    out = []
    for p in normalized:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def iso2_to_alpha3(code):
    """Convert ISO-2 code to ISO-3 code using pycountry. Return None if not mappable."""
    if not isinstance(code, str) or not code.strip():
        return None
    c = code.strip().upper()
    try:
        country = pycountry.countries.get(alpha_2=c)
        if country and hasattr(country, "alpha_3"):
            return country.alpha_3
    except Exception:
        pass
    # try alpha_3 given already
    try:
        country = pycountry.countries.get(alpha_3=c)
        if country and hasattr(country, "alpha_3"):
            return country.alpha_3
    except Exception:
        pass
    # fallback None
    return None

def iso2_to_name(code):
    if not isinstance(code, str) or not code.strip():
        return None
    c = code.strip().upper()
    try:
        country = pycountry.countries.get(alpha_2=c)
        if country:
            return country.name
    except Exception:
        pass
    try:
        country = pycountry.countries.get(alpha_3=c)
        if country:
            return country.name
    except Exception:
        pass
    # if already a name
    return code

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(CSV_PATH)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]  # drop unnamed cols

# ensure columns exist
expected_cols = {"title", "year", "doi", "cited_by", "journal", "concepts", "top_concept", "paper_country", "abstract", "authors"}
missing = expected_cols - set(df.columns)
if missing:
    print(f"Warning: missing expected columns (will continue): {missing}")

# coerce year, cited_by
df["year"] = pd.to_numeric(df.get("year", pd.Series()), errors="coerce")
df = df.dropna(subset=["year"]).copy()
df["year"] = df["year"].astype(int)

df["cited_by"] = pd.to_numeric(df.get("cited_by", pd.Series([0] * len(df))), errors="coerce").fillna(0).astype(int)

# fill missing lightly (we'll exclude Unknown/blanks from visuals)
df["paper_country"] = df.get("paper_country", pd.Series([""] * len(df))).fillna("")
df["language_mentioned"] = df.get("language_mentioned", pd.Series([""] * len(df))).fillna("")
df["journal"] = df.get("journal", pd.Series([""] * len(df))).fillna("")
df["title"] = df.get("title", pd.Series([""] * len(df))).fillna("")

# parse concepts (list-like) into list
df["concept_list"] = df.get("concepts", pd.Series([[]] * len(df))).apply(parse_concepts_field)

# parse authors into list of names
df["author_list"] = df.get("authors", pd.Series([[]] * len(df))).apply(parse_authors_field)

# normalize languages and explode
df["language_list"] = df["language_mentioned"].apply(normalize_languages_field)

# derive country_name and alpha3 for map; keep raw code for grouping if present
df["country_name"] = df["paper_country"].apply(lambda x: iso2_to_name(x) if isinstance(x, str) else None)
df["country_alpha3"] = df["paper_country"].apply(lambda x: iso2_to_alpha3(x) if isinstance(x, str) else None)

# exploded helper dfs (for counts & filtering)
df_lang = df.explode("language_list").copy()
df_lang["language_list"] = df_lang["language_list"].fillna("")

df_concepts = df.explode("concept_list").copy()
df_concepts["concept_list"] = df_concepts["concept_list"].fillna("")

df_authors = df.explode("author_list").copy()
df_authors["author_list"] = df_authors["author_list"].fillna("")

# -------------------------
# Dropdown options (exclude blank/Unknown)
# -------------------------
country_options = [{"label": n, "value": n} for n in sorted(df["country_name"].dropna().unique()) if str(n).strip()]
journal_options = [{"label": j, "value": j} for j in sorted(df["journal"].dropna().unique()) if str(j).strip()]
language_options = [{"label": l, "value": l} for l in sorted(df_lang["language_list"].dropna().unique()) if str(l).strip()]
concept_options = [{"label": c, "value": c} for c in sorted(df_concepts["concept_list"].dropna().unique()) if str(c).strip()]
author_options = [{"label": a, "value": a} for a in sorted(df_authors["author_list"].dropna().unique()) if str(a).strip()]

# -------------------------
# App layout (tabbed)
# -------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Create tabs with graph placeholders (so ids exist at layout time)
overview_tab = dbc.Tab(label="Overview", tab_id="tab_overview", children=[
    dbc.Row(dbc.Col(html.H5("Publications trend"))),
    dbc.Row(dbc.Col(dcc.Graph(id="overview_trend", style={"height": "400px"}))),
    dbc.Row(dbc.Col(html.H5("Publications per country (world map)"))),
    dbc.Row(dbc.Col(dcc.Graph(id="overview_map", style={"height": "600px"}))),
])

journals_countries_tab = dbc.Tab(label="Journals & Countries", tab_id="tab_journals_countries", children=[
    dbc.Row([dbc.Col(dcc.Graph(id="journal_bar", style={"height": "420px"}), width=6),
             dbc.Col(dcc.Graph(id="country_bar", style={"height": "420px"}), width=6)]),
    dbc.Row([dbc.Col(dcc.Graph(id="journal_trend", style={"height": "420px"}), width=6),
             dbc.Col(dcc.Graph(id="country_trend", style={"height": "420px"}), width=6)]),
])

langs_concepts_authors_tab = dbc.Tab(label="Languages, Concepts & Authors", tab_id="tab_langs_concepts_authors", children=[
    dbc.Row([dbc.Col(dcc.Graph(id="lang_bar", style={"height": "420px"}), width=6),
             dbc.Col(dcc.Graph(id="concept_bar", style={"height": "420px"}), width=6)]),
    dbc.Row([dbc.Col(dcc.Graph(id="lang_trend", style={"height": "420px"}), width=6),
             dbc.Col(dcc.Graph(id="concept_trend", style={"height": "420px"}), width=6)]),
    dbc.Row([dbc.Col(dcc.Graph(id="author_bar", style={"height": "420px"}), width=6),
             dbc.Col(dcc.Graph(id="author_citations", style={"height": "420px"}), width=6)]),
    dbc.Row([dbc.Col(dcc.Graph(id="author_trend", style={"height": "420px"}), width=12)]),
])

# Sidebar filters
controls = dbc.Card([
    dbc.CardBody([
        html.H5("Filters"),
        html.Label("Year range"),
        dcc.RangeSlider(
            id="year_range",
            min=int(df["year"].min()),
            max=int(df["year"].max()),
            step=1,
            value=[int(df["year"].min()), int(df["year"].max())],
            marks={int(y): str(int(y)) for y in sorted(df["year"].unique()) if int(y) % 5 == 0},
            tooltip={"placement": "bottom"}
        ),
        html.Br(),
        html.Label("Country"),
        dcc.Dropdown(id="country_filter", options=country_options, multi=True, placeholder="Filter by country"),
        html.Br(),
        html.Label("Journal"),
        dcc.Dropdown(id="journal_filter", options=journal_options, multi=True, placeholder="Filter by journal"),
        html.Br(),
        html.Label("Language"),
        dcc.Dropdown(id="language_filter", options=language_options, multi=True, placeholder="Filter by language"),
        html.Br(),
        html.Label("Concept"),
        dcc.Dropdown(id="concept_filter", options=concept_options, multi=True, placeholder="Filter by concept"),
        html.Br(),
        html.Label("Author"),
        dcc.Dropdown(id="author_filter", options=author_options, multi=True, placeholder="Filter by author"),
        html.Br(),
        html.Label("Title search"),
        dcc.Input(id="title_search", type="text", placeholder="Search titles", style={"width": "100%"}),
        html.Br(), html.Br(),
        dbc.Button("Reset filters", id="reset_btn", color="secondary", size="sm"),
        html.Br(), html.Br(),
        html.Div(id="selected_info", children="No selection", style={"fontSize": 12})
    ])
])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Linguistics publications — tabbed dashboard"), width=12)),
    dbc.Row([
        dbc.Col(controls, width=3),
        dbc.Col(dbc.Tabs([overview_tab, journals_countries_tab, langs_concepts_authors_tab], id="tabs", active_tab="tab_overview"), width=9)
    ]),
    dcc.Store(id="selection_store", data={"language": None, "journal": None, "country": None, "concept": None, "author": None})
], fluid=True)

# -------------------------
# Filtering helper
# -------------------------
def apply_filters(df_main, df_lang_exploded, df_concepts_exploded, df_authors_exploded,
                  year_range, countries, journals, languages, concepts, authors, title_search, selection_store):
    d = df_main.copy()
    # year range
    if year_range is None:
        year_range = [int(df["year"].min()), int(df["year"].max())]
    d = d[(d["year"] >= year_range[0]) & (d["year"] <= year_range[1])]

    # country filter (by country_name)
    if countries:
        d = d[d["country_name"].isin(countries)]

    # journals
    if journals:
        d = d[d["journal"].isin(journals)]

    # title search
    if title_search:
        p = str(title_search).lower()
        d = d[d["title"].fillna("").str.lower().str.contains(p)]

    # languages
    if languages:
        idx = df_lang_exploded[df_lang_exploded["language_list"].isin(languages)].index.unique()
        d = d.loc[d.index.isin(idx)]

    # concepts
    if concepts:
        idx = df_concepts_exploded[df_concepts_exploded["concept_list"].isin(concepts)].index.unique()
        d = d.loc[d.index.isin(idx)]

    # authors
    if authors:
        idx = df_authors_exploded[df_authors_exploded["author_list"].isin(authors)].index.unique()
        d = d.loc[d.index.isin(idx)]

    # selection_store cross-filter (click-to-filter)
    sel = selection_store or {}
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
    if sel.get("author"):
        idx = df_authors_exploded[df_authors_exploded["author_list"] == sel["author"]].index.unique()
        d = d.loc[d.index.isin(idx)]

    return d

# -------------------------
# Single update callback (renders every figure + table)
# -------------------------
@app.callback(
    Output("overview_trend", "figure"),
    Output("overview_map", "figure"),

    Output("journal_bar", "figure"),
    Output("journal_trend", "figure"),
    Output("country_bar", "figure"),
    Output("country_trend", "figure"),

    Output("lang_bar", "figure"),
    Output("lang_trend", "figure"),
    Output("concept_bar", "figure"),
    Output("concept_trend", "figure"),

    Output("author_bar", "figure"),
    Output("author_trend", "figure"),
    Output("author_citations", "figure"),

    Input("year_range", "value"),
    Input("country_filter", "value"),
    Input("journal_filter", "value"),
    Input("language_filter", "value"),
    Input("concept_filter", "value"),
    Input("author_filter", "value"),
    Input("title_search", "value"),
    Input("selection_store", "data"),
)
def update_all(year_range, countries, journals, languages, concepts, authors, title_search, selection_store):
    # compute filtered df
    dff = apply_filters(df, df_lang, df_concepts, df_authors, year_range, countries, journals, languages, concepts, authors, title_search, selection_store)

    # ---------------- overview trend
    pubs_by_year = dff.groupby("year").size().reset_index(name="count")
    fig_overview = px.line(pubs_by_year, x="year", y="count", title="Publications per year")
    fig_overview.update_traces(mode="lines+markers")

    # ---------------- overview world choropleth
    # count by country_alpha3 (require alpha-3 codes)
    map_df = dff[~dff["country_alpha3"].isna()].groupby("country_alpha3").size().reset_index(name="count")
    if not map_df.empty:
        fig_map = px.choropleth(map_df, locations="country_alpha3", color="count",
                                color_continuous_scale="Viridis", projection="natural earth",
                                labels={"count": "Publications"}, title="Publications per country")
    else:
        # empty placeholder
        fig_map = px.choropleth(locations=[], color=[], title="Publications per country")

    # Helper: function to make horizontal bar + top-N and trend for a given column
    def bar_and_trend_from_series(series, label, top_n=30, trend_top_n=TOP_N, series_name=None):
        """Return (bar_fig, trend_fig). series is a pandas Series of categories."""
        s = series.dropna()
        s = s[s.astype(str).str.strip() != ""]
        counts = s.value_counts().reset_index()
        counts.columns = [label, "count"]
        bar_df = counts.head(top_n)
        bar_fig = px.bar(bar_df, x="count", y=label, orientation="h", title=f"Top {label}")
        bar_fig.update_yaxes(autorange="reversed")

        top_trend = counts[label].head(trend_top_n).tolist()
        if top_trend:
            trend_df = dff[dff[series_name].isin(top_trend)].groupby(["year", series_name]).size().reset_index(name="count")
            trend_fig = px.line(trend_df, x="year", y="count", color=series_name, title=f"{label} trends (Top {trend_top_n})")
        else:
            trend_fig = px.line(title=f"{label} trends (Top {trend_top_n})")
        return bar_fig, trend_fig

    # --- Journals
    journal_bar_fig, journal_trend_fig = bar_and_trend_from_series(dff["journal"], "journal", top_n=30, trend_top_n=TOP_N, series_name="journal")

    # --- Countries
    country_bar_fig, country_trend_fig = bar_and_trend_from_series(dff["country_name"], "country", top_n=30, trend_top_n=TOP_N, series_name="country_name")

    # --- Languages (use exploded df)
    lang_series = df_lang.loc[dff.index]["language_list"]
    lang_bar_df = lang_series.dropna()
    lang_bar_df = lang_bar_df[lang_bar_df.astype(str).str.strip() != ""]
    lang_counts = lang_bar_df.value_counts().reset_index()
    lang_counts.columns = ["language", "count"]
    lang_counts = lang_counts.head(30)
    lang_bar_fig = px.bar(lang_counts.head(10), x="count", y="language", orientation="h", title="Top languages (top 10)")
    lang_bar_fig.update_yaxes(autorange="reversed")
    top_langs = lang_counts["language"].head(TOP_N).tolist()
    if top_langs:
        lt = df_lang.loc[dff.index]
        lt = lt[lt["language_list"].isin(top_langs)]
        lang_trend_df = lt.groupby(["year", "language_list"]).size().reset_index(name="count")
        lang_trend_fig = px.line(lang_trend_df, x="year", y="count", color="language_list", title=f"Language trends (Top {TOP_N})")
    else:
        lang_trend_fig = px.line(title=f"Language trends (Top {TOP_N})")

    # --- Concepts (from exploded df)
    concept_series = df_concepts.loc[dff.index]["concept_list"]
    concept_series = concept_series.dropna()
    concept_series = concept_series[concept_series.astype(str).str.strip() != ""]
    concept_counts = concept_series.value_counts().reset_index()
    concept_counts.columns = ["concept", "count"]
    concept_counts = concept_counts.head(30)
    concept_bar_fig = px.bar(concept_counts.head(8), x="count", y="concept", orientation="h", title="Top concepts (Top 8)")
    concept_bar_fig.update_yaxes(autorange="reversed")
    top_concepts = concept_counts["concept"].head(TOP_N).tolist()
    if top_concepts:
        ct = df_concepts.loc[dff.index]
        ct = ct[ct["concept_list"].isin(top_concepts)]
        concept_trend_df = ct.groupby(["year", "concept_list"]).size().reset_index(name="count")
        concept_trend_fig = px.line(concept_trend_df, x="year", y="count", color="concept_list", title=f"Concept trends (Top {TOP_N})")
    else:
        concept_trend_fig = px.line(title=f"Concept trends (Top {TOP_N})")

    # --- Authors: counts, trends, citations per author
    author_series = df_authors.loc[dff.index]["author_list"].dropna()
    author_series = author_series[author_series.astype(str).str.strip() != ""]
    author_counts = author_series.value_counts().reset_index()
    author_counts.columns = ["author", "count"]
    author_counts = author_counts.head(30)
    author_bar_fig = px.bar(author_counts.head(8), x="count", y="author", orientation="h", title="Top authors (Top 8)")
    author_bar_fig.update_yaxes(autorange="reversed")

    top_authors = author_counts["author"].head(TOP_N).tolist()
    if top_authors:
        at = df_authors.loc[dff.index]
        at = at[at["author_list"].isin(top_authors)]
        author_trend_df = at.groupby(["year", "author_list"]).size().reset_index(name="count")
        author_trend_fig = px.line(author_trend_df, x="year", y="count", color="author_list", title=f"Author trends (Top {TOP_N})")
        # citations per author: sum of cited_by across exploded matches
        # we need to associate each exploded author entry with its cited_by
        citations_df = at[["author_list", "cited_by"]].groupby("author_list").sum().reset_index()
        citations_df.columns = ["author", "total_citations"]
        citations_df = citations_df[citations_df["author"].isin(top_authors)]
        author_citations_fig = px.bar(citations_df, x="total_citations", y="author", orientation="h", title="Total citations per author (Top authors)")
        author_citations_fig.update_yaxes(autorange="reversed")
    else:
        author_trend_fig = px.line(title=f"Author trends (Top {TOP_N})")
        author_citations_fig = px.bar(title="Total citations per author (Top authors)")

    # Table for inspection (keep Unknown rows)
    table_df = dff[["year", "title", "journal", "cited_by", "paper_country", "doi"]].copy()
    columns = [{"name": c, "id": c} for c in table_df.columns]
    data = table_df.to_dict("records")

    return (fig_overview, fig_map,
            journal_bar_fig, journal_trend_fig, country_bar_fig, country_trend_fig,
            lang_bar_fig, lang_trend_fig, concept_bar_fig, concept_trend_fig,
            author_bar_fig, author_trend_fig, author_citations_fig,
            columns, data)

# -------------------------
# Combined click-to-filter + reset callback
# -------------------------
@app.callback(
    Output("selection_store", "data"),
    Output("year_range", "value"),
    Output("country_filter", "value"),
    Output("journal_filter", "value"),
    Output("language_filter", "value"),
    Output("concept_filter", "value"),
    Output("author_filter", "value"),
    Output("title_search", "value"),

    Input("lang_bar", "clickData"),
    Input("journal_bar", "clickData"),
    Input("country_bar", "clickData"),
    Input("concept_bar", "clickData"),
    Input("author_bar", "clickData"),
    Input("reset_btn", "n_clicks"),
    State("selection_store", "data"),
    prevent_initial_call=True
)
def update_filters(lang_click, journal_click, country_click, concept_click, author_click, reset_click, store):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    trig = ctx.triggered[0]["prop_id"].split(".")[0]
    store = store or {"language": None, "journal": None, "country": None, "concept": None, "author": None}

    if trig == "reset_btn":
        # reset everything
        return {"language": None, "journal": None, "country": None, "concept": None, "author": None}, \
               [int(df["year"].min()), int(df["year"].max())], None, None, None, None, None, ""

    if trig == "lang_bar" and lang_click:
        label = lang_click["points"][0].get("y") or lang_click["points"][0].get("label")
        return {"language": label, "journal": None, "country": None, "concept": None, "author": None}, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    if trig == "journal_bar" and journal_click:
        label = journal_click["points"][0].get("y") or journal_click["points"][0].get("label")
        return {"language": None, "journal": label, "country": None, "concept": None, "author": None}, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    if trig == "country_bar" and country_click:
        label = country_click["points"][0].get("y") or country_click["points"][0].get("label")
        return {"language": None, "journal": None, "country": label, "concept": None, "author": None}, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    if trig == "concept_bar" and concept_click:
        label = concept_click["points"][0].get("y") or concept_click["points"][0].get("label")
        return {"language": None, "journal": None, "country": None, "concept": label, "author": None}, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    if trig == "author_bar" and author_click:
        label = author_click["points"][0].get("y") or author_click["points"][0].get("label")
        return {"language": None, "journal": None, "country": None, "concept": None, "author": label}, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    return store, no_update, no_update, no_update, no_update, no_update, no_update, no_update

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    print("Starting dashboard on http://127.0.0.1:8050")
    app.run(debug=True)
