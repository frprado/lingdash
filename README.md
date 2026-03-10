# Lingdash

A bibliometric dashboard tracking peer-reviewed linguistics publications indexed in [OpenAlex](https://openalex.org). It maps where linguistics research comes from, which languages are studied most, and how the breadth of the field's agenda has changed over time.

![Report preview](preview.png)

## What it does

- Fetches linguistics papers from the OpenAlex API (subfield 1203)
- Filters to qualified journals (DOAJ, OpenAlex core, or h-index ≥ 15)
- Extracts languages mentioned in titles/abstracts via regex
- Classifies papers by language family
- Renders an interactive [Dash](https://dash.plotly.com) dashboard and a self-contained HTML/PDF report

## Structure

```
pipeline/
├── extraction.py            # fetch papers from OpenAlex → data.parquet
├── build_journal_allowlist.py  # build journal quality filter
├── add_language_families.py    # enrich linglist.csv with language families
├── classify_subfields.py       # subfield classification
├── dashboard.py             # interactive Dash app
├── generate_report.py       # static HTML / PDF / PNG report
├── linglist.csv             # reference list of language names
└── data.parquet             # extracted dataset
```

## Usage

```bash
pip install -r requirements.txt

# Run the dashboard
python pipeline/dashboard.py

# Generate the report
python pipeline/generate_report.py          # HTML
python pipeline/generate_report.py --pdf    # + PDF  (requires playwright)
python pipeline/generate_report.py --png    # + PNG  (requires playwright)
```

> **Note:** Scripts that call the OpenAlex API require you to set your email in the `User-Agent` header (see the `mailto:YOUR_EMAIL_HERE` placeholder). This is part of OpenAlex's [polite pool](https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication) policy.

---

By [Frederico Prado](https://frprado.github.io)
