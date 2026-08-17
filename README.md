# RAG Search Engine

A search engine built as part of the [Retrieval Augmented Generation course](https://www.boot.dev/courses/learn-retrieval-augmented-generation) on Boot.dev.

## About

This project is a hands-on implementation of a RAG (Retrieval Augmented Generation) system in Python, covering keyword search, tokenization, stop words, stemming, vector embeddings, and more.

The dataset is a collection of 5.000 movies (`data/movies.json`), each with a title and a description.

## Setup

```bash
uv sync
```

## Usage

All commands run through the CLI entrypoint:

```bash
uv run cli/keyword_search_cli.py <command> [args]
```

### `build`

Builds the inverted index from the dataset and caches it to disk (`cache/`).
Run this first — the other commands load the index from that cache.

```bash
uv run cli/keyword_search_cli.py build
```

```
Index built successfully!
```

### `search`

Searches movies by keyword and returns up to 5 matches.

```bash
uv run cli/keyword_search_cli.py search "cyborg"
```

```
Searching for: cyborg
1. Superman: Unbound (ID 229)
2. Code Name: S.T.E.A.M. (ID 250)
3. Highlander: The Animated Series (ID 421)
4. Eliminators (ID 638)
5. Warrior of the Lost World (ID 950)
```

### `tf`

Term frequency — how many times a term appears **in a single document**.

```bash
uv run cli/keyword_search_cli.py tf 1 police
```

```
6
```

### `idf`

Inverse document frequency — how rare a term is **across the whole collection**.
Common terms score low, rare terms score high.

```bash
uv run cli/keyword_search_cli.py idf cyborg
```

```
Inverse document frequency of 'cyborg': 5.34
```

```bash
uv run cli/keyword_search_cli.py idf police
```

```
Inverse document frequency of 'police': 1.14
```

### `tfidf`

TF-IDF — the product of the two scores above. It answers: *how important is this
term to this specific document, relative to the whole collection?*

A term scores high only when it appears often in the document **and** is rare
elsewhere. Compare two terms in the same movie:

```bash
uv run cli/keyword_search_cli.py tfidf 1 police
```

```
TF-IDF score of 'police' in document '1': 6.87
```

```bash
uv run cli/keyword_search_cli.py tfidf 1 anbuselvan
```

```
TF-IDF score of 'anbuselvan' in document '1': 140.84
```

Both terms appear in *Kaakha..Kaakha: The Police*, but `anbuselvan` (the
protagonist) is far more distinctive: it appears 18 times here and almost
nowhere else in the dataset, while `police` shows up in 1591 of the 5000 movies.

### `bm25idf`

BM25 IDF — a refined version of the IDF above, and the first of three
improvements BM25 makes over plain TF-IDF.

```bash
uv run cli/keyword_search_cli.py bm25idf cyborg
```

```
BM25 IDF score of 'cyborg': 5.36
```

```bash
uv run cli/keyword_search_cli.py bm25idf police
```

```
BM25 IDF score of 'police': 1.14
```

For ordinary terms the two formulas land in almost the same place — compare
these to the `idf` values above. They only diverge at the extremes: the BM25
variant is derived from a probabilistic relevance model and stays well behaved
for terms that appear in most of the collection, where the plain formula would
otherwise collapse toward zero.

## How it works

Queries and documents go through the same pipeline before being compared:
lowercasing → punctuation removal → stop word filtering (`data/stopwords.txt`) → stemming (Porter).

The index is stored as three pickled structures under `cache/`:

| File | Structure | Purpose |
|---|---|---|
| `index.pkl` | `token → set of doc ids` | inverted index, drives lookups |
| `docmap.pkl` | `doc id → movie` | retrieves the full record for a result |
| `term_frequencies.pkl` | `doc id → Counter(token → count)` | term counts, used for scoring |
