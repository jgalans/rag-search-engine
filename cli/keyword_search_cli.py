import argparse
import json
import string
"""PICKLE: Es una librería de Python que convierte cualquier objeto Python 
(diccionarios, listas, clases...) en bytes para guardarlo en disco
 y recuperarlo después. Es como "congelar" el objeto."""
import pickle
import os
from nltk.stem import PorterStemmer
from collections import Counter

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

def tokenize_term(term: str, stopwords: list[str] = []) -> str:
    tokens = tokenize(term, stopwords)
    if len(tokens) != 1:
        raise ValueError(f"Expected exactly one token, got {len(tokens)} for term: '{term}'")
    return tokens[0]

class InvertedIndex:
    def __init__(self, stopwords: list[str] = []):
        self.index = {}
        self.docmap = {}
        self.stopwords = stopwords
        self.term_frequencies = {}  # doc_id → Counter de tokens

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text, self.stopwords)
        # Asegura si hay un Counter para este doc_id en term_frequencies.
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        self.term_frequencies[doc_id].update(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
        # Guarda el conteo de tokens para este documento
        self.term_frequencies[doc_id] = Counter(tokens)

    def get_documents(self, term: str) -> list[int]:
        term = term.lower()
        return sorted(self.index.get(term, set()))
    
    def build(self, movies: list[dict]) -> None:
        for movie in movies:
            # Saca el ID de la película y lo guarda en una variable
            doc_id = movie["id"]
            # Concatena el título y la descripción en un solo texto para indexarlo
            text = f"{movie['title']} {movie['description']}"
            # Guarda la película completa en el docmap, usando su ID como clave
            self.docmap[doc_id] = movie
            # Tokeniza el texto y añade el ID al índice por cada token
            self.__add_document(doc_id, text)
        
    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:    
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:    
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:    
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        if not os.path.exists("cache/index.pkl") or not os.path.exists("cache/docmap.pkl") or not os.path.exists("cache/term_frequencies.pkl"): 
            raise FileNotFoundError("Index files not found. Run 'build' first.")
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:    
            self.term_frequencies = pickle.load(f)

    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id].get(term, 0)

def main() -> None:
    with open("data/movies.json", "r") as f:
        movies = json.load(f)

    with open("data/stopwords.txt", "r") as f:
        stopwords = f.read().splitlines()

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    #search
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    #build
    subparsers.add_parser("build", help="Build the inverted index")

    #tf
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to look up")

    args = parser.parse_args()

    match args.command:
        case "search":
            index = InvertedIndex(stopwords)
            try:
                index.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return
            
            results = []
            tokens = tokenize(args.query, stopwords)

            for token in tokens:
                for doc_id in index.get_documents(token):
                    if doc_id not in [r["id"] for r in results]:  # Evita duplicados
                        results.append(index.docmap[doc_id])
                    if len(results) >= 5:  # Limita a 5 resultados
                        break
                if len(results) >= 5:
                    break

            print(f"Searching for: {args.query}")
            for i, movie in enumerate(results, start=1):
                print(f"{i}. {movie['title']} (ID {movie['id']})")

        case "build":
            index = InvertedIndex(stopwords)
            index.build(movies["movies"])
            index.save()
            print("Index built successfully!")
            #docs = index.get_documents("merida")
            #print(f"First document for token 'merida' = {docs[0]}")

        case "tf":
            index = InvertedIndex(stopwords)
            try:
                index.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return
            term = tokenize_term(args.term, stopwords)
            tf = index.get_tf(args.doc_id, term)
            print(tf)

        case _:
            parser.print_help()

"""
De esta forma si importamos este archivo en otro lugar, 
no se ejecutará el main() automáticamente, 
sino que solo se ejecutará cuando se ejecute este archivo directamente.
"""
if __name__ == "__main__":
    main()