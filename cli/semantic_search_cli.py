import argparse
from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticSearch:
    """Semantic search class"""
    def __init__(self):
        """Initializes the model"""
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text: str) -> np.ndarray:
        """Genera el embedding de un texto.

        `encode` trabaja por lotes: recibe una lista y devuelve una matriz
        (n, 384). Como aquí solo hay un texto, se toma la primera fila para
        devolver un único vector de 384 dimensiones.
        """
        if not text.strip():
            raise ValueError("Text is empty or contains only whitespace.")
        return self.model.encode([text])[0] #le paso la text en una lista directamente como argumento [text]


def verify_model() -> None:
    """Create the model and print its information"""
    search = SemanticSearch()

    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def embed_text(text: str) -> None:
    """Imprime el embedding de un texto: su inicio y su número de dimensiones."""
    ssearch = SemanticSearch()
    embedding = ssearch.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    #verify
    subparsers.add_parser("verify", help="Verify the model")

    #embed_text
    embed_text_parser = subparsers.add_parser("embed_text", help="Get the embedding for a given text")
    embed_text_parser.add_argument("text", type=str, help="Text to embedding")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()