import argparse
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    """Semantic search class"""
    def __init__(self):
        """Initializes the model"""
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

def verify_model() -> None:
    """Create the model and print its information"""
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    #verify
    subparsers.add_parser("verify", help="Verify the model")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()