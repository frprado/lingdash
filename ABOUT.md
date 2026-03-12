# Linguistics Research — A Bibliometric Study

## Data Source & Scope

**Source:** [OpenAlex](https://openalex.org), a fully open bibliographic index covering ~250 million scholarly works.

**Corpus:** Peer-reviewed journal articles tagged to linguistics topics under OpenAlex subfield 1203, spanning 1960–2024. Five topic IDs were targeted to capture the breadth of linguistics research:

| Topic ID | Label |
|----------|-------|
| T10034 | Syntax, Semantics, Linguistic Variation |
| T12373 | Linguistic research and analysis |
| T13156 | Historical Linguistics and Language Studies |
| T13538 | Linguistics and Language Analysis |
| T13260 | Linguistics and Language Studies |

Papers were retrieved sorted by citation count descending, using cursor-based pagination to bypass OpenAlex's 10 000-record page limit. Only works of type `article` were included; books, book chapters, and preprints were excluded.

---

## Filtering & Quality Control

Raw OpenAlex results include a long tail of low-quality venues — predatory journals, single-country local publications, and conference proceedings misclassified as journal articles. A journal allowlist was built to filter these out.

**A journal qualifies if it meets any one of:**

1. **DOAJ membership** (`is_in_doaj:true`) — editorial vetting by the Directory of Open Access Journals
2. **OpenAlex core literature** (`is_core:true`) — included in OpenAlex's curated core set
3. **h-index ≥ 15** — has produced a sustained body of cited work over time

The three criteria were queried separately from the OpenAlex Sources endpoint (~500 paginated requests) and merged into a normalised name lookup. Journal names were normalised before matching: lowercased, punctuation replaced by spaces, and whitespace collapsed. Papers published in journals with no match in the allowlist were dropped.

This filter is **country-agnostic**: it removes marginal venues globally, not just in any one region.

---

## Data Cleaning & Enrichment

### Abstract reconstruction

OpenAlex stores abstracts as inverted indexes (a mapping of `word → [positions]`) to avoid copyright issues. Abstracts were reconstructed by inverting the index: each word was placed at its positions and the sequence was joined into prose. Papers with no abstract were retained but left blank for downstream text processing.

### Country attribution

Country of origin was assigned to the **first author's first institutional affiliation** that carried a country code. Papers with no resolvable affiliation were left as missing and excluded from geographic analyses.

### Language extraction

A reference list of ~7 700 language names (`linglist.csv`) was compiled from ISO 639 codes and their standard display names. A regex pattern was built from this list and applied to the concatenated title + abstract of each paper. All matching language names were deduplicated (case-insensitive) and stored per paper. A paper with no language mention was left blank rather than assigned a default.

### Language family classification

Each language in the reference list was assigned to a top-level genetic family through a three-tier lookup:

1. **Hardcoded ISO 639-2 → family table** — covers ~400 codes for the most common language families (Indo-European, Sino-Tibetan, Afro-Asiatic, Niger-Congo, Austronesian, Dravidian, Turkic, and ~20 others). Fast and offline.
2. **Wikidata SPARQL fallback** — for codes not in the hardcoded table, a SPARQL query traverses the Wikidata `P279` (subclass-of) chain up to the root family node (`Q25295`) and maps the result to a normalised family label.
3. **Name-pattern fallback** — a final regex pass on the language's display name catches any codes Wikidata could not resolve (e.g. umbrella or uncoded entries).

Umbrella codes (`mul`, `mis`, `zxx`) and unclassifiable entries were tagged `Uncoded` or `Other/Unclassified` and excluded from family visualisations.

### Subfield classification

Each paper was assigned to one of twelve linguistics subfields using zero-shot embedding similarity:

- Title + first 300 characters of abstract were embedded with `all-MiniLM-L6-v2` (sentence-transformers, CPU-friendly, ~90 MB).
- Twelve subfield descriptions were assembled as bags of representative vocabulary — not prose — so the model embeds them into the correct region of semantic space.
- Cosine similarity (dot product of L2-normalised vectors) was computed and the closest subfield was assigned.

Subfields: Phonology & Phonetics · Syntax & Grammar · Semantics · Pragmatics & Discourse · Sociolinguistics · Historical & Comparative Linguistics · Language Acquisition · Psycholinguistics & Neurolinguistics · Computational Linguistics & NLP · Applied Linguistics & Language Teaching · Morphology · Language Documentation & Typology.

---

## Key Findings

### Output growth
Annual publication volume grew substantially from the 1960s to the 2020s — a trend that reflects both genuine expansion of the field and the progressive digitisation of scholarly publishing that has made older work indexable.

### Geographic concentration
Research output is heavily concentrated in a small number of countries. A single country accounts for a disproportionate share of the corpus, and this imbalance persists across decades, though the relative shares of the top producers have shifted over time.

### Language bias
English dominates as the object of study by a wide margin. Indo-European languages collectively account for the large majority of language mentions. Languages from Austronesian, Niger-Congo, and Afro-Asiatic families — which together encompass the majority of the world's speakers — are systematically underrepresented.

### Diversity over time
Shannon entropy (H bits) computed per year measures how evenly linguistic attention is distributed. A higher H indicates that more languages are studied in roughly equal proportions in that year; a lower H indicates concentration on a few languages. The diversity index shows modest growth over the study period, but the field's linguistic gaze remains narrow relative to the world's actual linguistic diversity.

### Journals as gatekeepers
Language diversity varies substantially across journals. Some venues consistently publish research on a wide range of languages; others are effectively monolingual in their object of study. This heterogeneity points to editorial policy — and the communities journals serve — as a structural factor in shaping which languages get academic attention.

---

## Limitations

- **Language detection is lexical.** The regex approach matches language names in text but cannot distinguish a paper *about* Swahili from one that merely *mentions* it in passing. Precision is high for papers whose titles name the language studied; recall is lower for papers that describe a language only implicitly (e.g. "we analysed a corpus from Kenya").
- **First-author country is a proxy.** Multi-institutional and cross-national collaborations are attributed to a single country. Papers with no institutional affiliation data are dropped entirely.
- **OpenAlex topic tags are imperfect.** The five topic IDs used are a reasonable approximation of linguistics, but the corpus will include some non-linguistics papers and exclude some linguistics papers tagged to adjacent fields.
- **Citation sorting may introduce recency bias.** Fetching in descending citation order means that very recent papers (with little time to accumulate citations) are underrepresented in any truncated extraction.

---

## Running the Analysis

```bash
pip install -r requirements.txt

# 1. Fetch papers from OpenAlex
python pipeline/extraction.py

# 2. Build the journal quality filter
python pipeline/build_journal_allowlist.py

# 3. Enrich with language families
python pipeline/add_language_families.py

# 4. Classify papers by subfield
python pipeline/classify_subfields.py

# 5. Launch the interactive dashboard
python pipeline/dashboard.py

# 6. Generate the static report
python pipeline/generate_report.py          # HTML
python pipeline/generate_report.py --pdf    # + PDF (requires playwright)
python pipeline/generate_report.py --png    # + PNG (requires playwright)
```

> **Note:** Scripts that call external APIs require your email in the `User-Agent` / `HEADERS` fields (see the `mailto:YOUR_EMAIL_HERE` placeholder). This is part of OpenAlex's [polite pool](https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication) policy.

---

By [Frederico Prado](https://frprado.github.io)
