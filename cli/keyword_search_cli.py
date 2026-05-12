import argparse
import json
import string
from nltk.stem import PorterStemmer

"""Se crea un objeto de la clase PorterStemmer para usarlo en la función matches,
    que se encarga de comparar los tokens de la consulta con los tokens del título de la película."""
stemmer = PorterStemmer()


def remove_punctuation(text: str) -> str:
    """string.punctuation contiene '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'"""
    return text.translate(str.maketrans("", "", string.punctuation))

def tokenize(text: str, stopwords: list[str] = []) -> list[str]:
    text = remove_punctuation(text.lower())
    tokens = text.split()
    tokens = [token for token in tokens if token and token not in stopwords]  # elimina tokens vacíos
    tokens = [stemmer.stem(token) for token in tokens]  # aplica stemming a cada token
    return tokens

def matches(query: str, title: str, stopwords: list[str]) -> bool:
    query_tokens = tokenize(query, stopwords)
    title_tokens = tokenize(title, stopwords)

    return any(
        query_token in title_token
        for query_token in query_tokens
        for title_token in title_tokens
    )

def main() -> None:
    with open("data/movies.json", "r") as f:
        movies = json.load(f)

    with open("data/stopwords.txt", "r") as f:
        stopwords = f.read().splitlines()

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            results = []
            for movie in movies["movies"]:
                if matches(args.query, movie["title"], stopwords):
                    results.append(movie)

            results = results[:5]  # truncar a 5 resultados

            print(f"Searching for: {args.query}")
            for i, movie in enumerate(results, start=1):
                print(f"{i}. {movie['title']}")
        case _:
            parser.print_help()

"""
De esta forma si importamos este archivo en otro lugar, 
no se ejecutará el main() automáticamente, 
sino que solo se ejecutará cuando se ejecute este archivo directamente.
"""
if __name__ == "__main__":
    main()