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

### `bm25tf`

BM25 TF — a *saturating*, length-normalized version of term frequency. It is the
second and third improvement BM25 makes over plain TF-IDF, in one formula:

```
tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))
```

Raw TF grows without limit: a term appearing 100 times scores 100. BM25 instead
rises quickly for the first few occurrences and then flattens out, approaching a
ceiling of `k1 + 1`. The `b` term corrects for document length, so a long
synopsis does not score highly just by having more room for words.

```bash
uv run cli/keyword_search_cli.py bm25tf 1 police
```

```
BM25 TF score of 'police' in document '1': 2.09
```

```bash
uv run cli/keyword_search_cli.py bm25tf 1 anbuselvan
```

```
BM25 TF score of 'anbuselvan' in document '1': 2.35
```

Compare these to the raw counts: `police` appears 6 times in that movie and
`anbuselvan` 18 — three times as often — yet the scores are 2.09 and 2.35. The
saturation curve for that document looks like this:

| raw tf | 1 | 2 | 3 | 6 | 18 | 100 |
|---|---|---|---|---|---|---|
| BM25 tf | 1.15 | 1.57 | 1.79 | 2.09 | 2.35 | 2.47 |

The intuition: the difference between a term appearing once and twice is
meaningful, while the difference between 50 and 100 times is not. `k1` controls
how fast the curve saturates (default `1.5`), and `b` how strongly document
length is corrected for (default `0.75`, where `0` disables it entirely). Both
can be passed as optional arguments.

### `bm25search`

Full BM25 ranking — the three improvements above, applied to a real search.

A document's score is the **sum** of its BM25 over every token in the query, so
matching more query terms ranks higher. Only documents containing at least one
query token are scored; the rest never enter the ranking.

```bash
uv run cli/keyword_search_cli.py bm25search "animated family"
```

```
1. (2929) Gakuen Alice - Score: 7.35
2. (2275) Day of the Animals - Score: 7.13
3. (1907) Fantastic Mr. Fox - Score: 6.92
4. (1125) The Jungle Book - Score: 6.77
5. (2665) Sing - Score: 6.75
```

```bash
uv run cli/keyword_search_cli.py bm25search "cyborg police"
```

```
1. (638) Eliminators - Score: 11.35
2. (2660) Metal Gear Rising: Revengeance - Score: 11.13
3. (4941) Justice League: Throne of Atlantis - Score: 10.97
4. (1267) Justice League vs. Teen Titans - Score: 10.70
5. (4443) Justice League: The Flashpoint Paradox - Score: 10.03
```

Contrast this with `search`, which returns the first five documents containing
any query token, in index order. Same index, same tokens — but `bm25search`
answers "which of these are most relevant?" instead of "which of these match?".

## How it works

Queries and documents go through the same pipeline before being compared:
lowercasing → punctuation removal → stop word filtering (`data/stopwords.txt`) → stemming (Porter).

The stop word list is put through the same lowercasing and punctuation removal as
the text itself. Without that step, contractions like `don't` in the list never
match the `dont` produced by the tokenizer, and ~45 stop words silently survive
into every document — inflating document lengths and skewing BM25's length
normalization.

The index is stored as three pickled structures under `cache/`:

| File | Structure | Purpose |
|---|---|---|
| `index.pkl` | `token → set of doc ids` | inverted index, drives lookups |
| `docmap.pkl` | `doc id → movie` | retrieves the full record for a result |
| `term_frequencies.pkl` | `doc id → Counter(token → count)` | term counts, used for scoring |
| `doc_lengths.pkl` | `doc id → token count` | BM25 length normalization |
