"""
classify_subfields.py

Assigns each paper in data.csv to a linguistics subfield using zero-shot
embedding similarity (sentence-transformers/all-MiniLM-L6-v2, CPU-friendly).

Each paper's title + first 300 chars of abstract are embedded and compared
to rich descriptions of each subfield. The closest subfield wins.

Reads:  pipeline/data.csv
Writes: pipeline/data.csv  (adds/overwrites the 'subfield' column)

Usage:
    python pipeline/classify_subfields.py
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

_DIR = os.path.dirname(__file__)

# ── Taxonomy ──────────────────────────────────────────────────────────────────
# Descriptions are bags of representative vocabulary, not prose, so the model
# embeds them in the right region of semantic space.
SUBFIELDS = {
    "Phonology & Phonetics": (
        "phonology phonetics sounds speech articulation prosody intonation tone "
        "syllable consonant vowel phoneme acoustic phonological optimality theory "
        "minimal pair assimilation vowel harmony stress pitch"
    ),
    "Syntax & Grammar": (
        "syntax grammar sentence structure constituency dependency parse tree "
        "clause phrase verb argument structure generative transformational grammar "
        "subject object movement island wh- relative clause binding"
    ),
    "Semantics": (
        "semantics meaning reference truth conditions lexical semantics "
        "compositionality entailment word sense ambiguity metaphor polysemy "
        "formal semantics quantifier scope tense aspect modality"
    ),
    "Pragmatics & Discourse": (
        "pragmatics discourse conversation implicature speech act coherence "
        "cohesion information structure topic focus context utterance "
        "relevance theory politeness face hedging anaphora reference"
    ),
    "Sociolinguistics": (
        "sociolinguistics language variation dialect social identity code-switching "
        "bilingualism multilingualism language contact community stigma register "
        "style gender ethnicity language attitudes language policy planning"
    ),
    "Historical & Comparative Linguistics": (
        "historical linguistics language change diachronic etymology reconstruction "
        "proto-language comparative method cognate sound change language family "
        "Indo-European Slavic Romance Germanic language relatedness"
    ),
    "Language Acquisition": (
        "language acquisition first language second language L2 child language "
        "learning development input grammar learnability bilingual acquisition "
        "critical period age of acquisition vocabulary growth"
    ),
    "Psycholinguistics & Neurolinguistics": (
        "psycholinguistics neurolinguistics processing reading eye tracking "
        "aphasia brain EEG fMRI cognitive reaction time working memory "
        "sentence comprehension production priming lexical access"
    ),
    "Computational Linguistics & NLP": (
        "computational linguistics NLP natural language processing corpus annotation "
        "parsing machine translation word embeddings neural language model "
        "sentiment analysis named entity tagging dependency parsing"
    ),
    "Applied Linguistics & Language Teaching": (
        "applied linguistics language teaching pedagogy EFL ESL curriculum "
        "classroom instruction foreign language education TESOL writing "
        "literacy reading comprehension language skills assessment"
    ),
    "Morphology": (
        "morphology inflection derivation word formation morpheme stem affix "
        "paradigm nominal verbal agreement case gender number plural "
        "compounding cliticization reduplication"
    ),
    "Language Documentation & Typology": (
        "language documentation typology linguistic diversity endangered language "
        "fieldwork descriptive grammar universal cross-linguistic variation "
        "minority language revitalization corpus building"
    ),
}

LABELS       = list(SUBFIELDS.keys())
DESCRIPTIONS = list(SUBFIELDS.values())

# ── Load data ─────────────────────────────────────────────────────────────────
csv_path = os.path.join(_DIR, "data.csv")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df):,} rows.")

# Title + first 300 chars of abstract gives enough signal without being slow
texts = (
    df["title"].fillna("") + " " +
    df["abstract"].fillna("").str[:300]
).tolist()

# ── Embed ─────────────────────────────────────────────────────────────────────
print("Loading model (downloads ~90 MB on first run)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding subfield descriptions...")
desc_embs = model.encode(DESCRIPTIONS, normalize_embeddings=True, show_progress_bar=False)

print(f"Embedding {len(texts):,} papers — this may take 5–15 min on CPU...")
text_embs = model.encode(
    texts,
    batch_size=256,
    normalize_embeddings=True,
    show_progress_bar=True,
)

# ── Assign closest subfield ───────────────────────────────────────────────────
# normalized embeddings → cosine similarity = dot product
sims = text_embs @ desc_embs.T          # (N, n_subfields)
best = np.argmax(sims, axis=1)
df["subfield"] = [LABELS[i] for i in best]

print("\nSubfield distribution:")
print(df["subfield"].value_counts().to_string())

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(csv_path, index=False)
print(f"\nSaved → {csv_path}")
