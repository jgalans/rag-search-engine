Los bloques de código no se están renderizando bien porque puse las barras invertidas `\` para escaparlos en mi respuesta, pero se copiaron tal cual. 

Edita el README en GitHub y reemplaza el contenido por esto exactamente:

````markdown
# RAG Search Engine

A search engine built as part of the [Retrieval Augmented Generation course](https://www.boot.dev/courses/build-a-rag-search-engine-python) on Boot.dev.

## About

This project is a hands-on implementation of a RAG (Retrieval Augmented Generation) system in Python, covering keyword search, tokenization, stop words, stemming, vector embeddings, and more.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run cli/keyword_search_cli.py search "your query here"
```
````
