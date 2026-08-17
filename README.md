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

<pre>
Index built successfully!
</pre>

### `search`

Searches movies by keyword and returns up to 5 matches.

```bash
uv run cli/keyword_search_cli.py search "cyborg"
```

<pre>
Searching for: cyborg
1. Superman: Unbound (ID 229)
2. Code Name: S.T.E.A.M. (ID 250)
3. Highlander: The Animated Series (ID 421)
4. Eliminators (ID 638)
5. Warrior of the Lost World (ID 950)
</pre>

### `tf`

Term frequency — how many times a term appears **in a single document**.

```bash
uv run cli/keyword_search_cli.py tf 1 police
```

<pre>
6
</pre>

### `idf`

Inverse document frequency — how rare a term is **across the whole collection**.
Common terms score low, rare terms score high.

```bash
uv run cli/keyword_search_cli.py idf cyborg
```

<pre>
Inverse document frequency of 'cyborg': 5.34
</pre>

```bash
uv run cli/keyword_search_cli.py idf police
```

<pre>
Inverse document frequency of 'police': 1.14
</pre>

### `tfidf`

TF-IDF — the product of the two scores above. It answers: *how important is this
term to this specific document, relative to the whole collection?*

A term scores high only when it appears often in the document **and** is rare
elsewhere. Compare two terms in the same movie:

```bash
uv run cli/keyword_search_cli.py tfidf 1 police
```

<pre>
TF-IDF score of 'police' in document '1': 6.87
</pre>

```bash
uv run cli/keyword_search_cli.py tfidf 1 anbuselvan
```

<pre>
TF-IDF score of 'anbuselvan' in document '1': 140.84
</pre>

Both terms appear in *Kaakha..Kaakha: The Police*, but `anbuselvan` (the
protagonist) is far more distinctive: it appears 18 times here and almost
nowhere else in the dataset, while `police` shows up in 1591 of the 5000 movies.

## How it works

Queries and documents go through the same pipeline before being compared:
lowercasing → punctuation removal → stop word filtering (`data/stopwords.txt`) → stemming (Porter).

The index is stored as three pickled structures under `cache/`:

| File | Structure | Purpose |
|---|---|---|
| `index.pkl` | `token → set of doc ids` | inverted index, drives lookups |
| `docmap.pkl` | `doc id → movie` | retrieves the full record for a result |
| `term_frequencies.pkl` | `doc id → Counter(token → count)` | term counts, used for scoring |
