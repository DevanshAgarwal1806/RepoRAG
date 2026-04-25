import json
import re
from rank_bm25 import BM25Okapi
from retriever.query_expansion import expand_query 
import os

# 1. Data Loader
def load_corpus(filepath: str) -> list[dict]:
    """Loads the parsed codebase from a JSON file."""
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        return []

# 2. Code-Specific Tokenization
def tokenize_code(text: str) -> list[str]:
    """
    Advanced tokenizer for code.
    Splits camelCase and snake_case so BM25 can match individual words.
    """
    if not text:
        return []
        
    # Split camelCase (e.g., 'connectDb' -> 'connect Db')
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Replace underscores with spaces (e.g., 'authenticate_user' -> 'authenticate user')
    text = text.replace('_', ' ')
    
    # Replace common code punctuation with spaces to isolate words
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Lowercase and split into a final list of tokens
    return text.lower().split()

def bm25_search(query: str, corpus: list[dict], top_k: int = 3) -> list[dict]:
    """Performs BM25 search on the corpus given a user query."""
    # Tokenize the corpus
    tokenized_corpus = [tokenize_code(doc["source_code"]) for doc in corpus]
    
    # Initialize BM25
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Tokenize the expanded query
    tokenized_query = tokenize_code(query)
    
    # Get BM25 scores
    scores = bm25.get_scores(tokenized_query)
    
    # Zip scores with documents and sort by score in descending order
    results = sorted(zip(scores, corpus), key=lambda x: x[0], reverse=True)
    
    # Return top_k results
    return [(doc, score) for score, doc in results[:top_k]]

# 3. Main Search Pipeline
if __name__ == "__main__":
    # 1. Load the codebase
    BASE_DIR = os.path.dirname(__file__)
    corpus_path = os.path.normpath(
        os.path.join(BASE_DIR, "..", "sample_repository_output", "extracted_functions.json")
    )
    corpus = load_corpus(corpus_path)
    if not corpus:
        exit()

    user_query = "how to connect to the database"
    print(f"Original Query : {user_query}")
    
    # 2. Expand the query using an LLM (optional but can boost recall)
    expanded_query = expand_query(user_query)
    
    # 2. Perform BM25 search
    results = bm25_search(expanded_query, corpus, top_k=3)
    
    print("--- Top Ranked Search Results ---")
    # Display the top 3 results (Recall@3)
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"Rank {rank} | Function: {doc['id']} | Score: {score}")