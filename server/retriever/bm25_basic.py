import json
import re
from rank_bm25 import BM25Okapi
from query_expansion import expand_query 
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

    # 2. Tokenize the entire codebase
    # We apply the tokenize_code function to the "source_code" field of every document
    tokenized_corpus = [tokenize_code(doc["source_code"]) for doc in corpus]

    # 3. Initialize the BM25 Index
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"Successfully indexed {len(corpus)} functions.\n")

    # 4. User Interaction
    user_query = "how to connect to the database"
    print(f"Original Query : {user_query}")
    
    # 5. Expand the Query (Bridging the semantic gap)
    expanded_query = expand_query(user_query)
    print(f"Expanded Query : {expanded_query}\n")
    
    # 6. Tokenize the expanded query
    tokenized_query = tokenize_code(expanded_query)
    
    # 7. Get BM25 scores
    scores = bm25.get_scores(tokenized_query)
    
    # 8. Rank and Display Results
    # Zip the scores with the original documents and sort them in descending order
    results = sorted(zip(scores, corpus), key=lambda x: x[0], reverse=True)
    
    print("--- Top Ranked Search Results ---")
    # Display the top 3 results (Recall@3)
    for rank, (score, doc) in enumerate(results[:3], start=1):
        print(f"Rank {rank} | Score: {score:.4f} | Function: {doc['name']}")