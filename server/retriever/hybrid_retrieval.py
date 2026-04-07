import warnings
import json
import os
from rank_bm25 import BM25Okapi

# Import your modular components
from query_expansion import expand_query
from bm25_basic import tokenize_code # Reusing your regex tokenizer
from dense_retrieval import get_dense_rankings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# RRF Formula Logic
# ---------------------------------------------------------
def calculate_rrf(bm25_ranks: dict, dense_ranks: dict, k: int = 60) -> list[tuple[str, float]]:
    """
    Combines ranks using Reciprocal Rank Fusion.
    Formula: RRF_Score = sum( 1 / (k + rank) )
    """
    rrf_scores = {}
    
    # Get a unique set of all document IDs from both lists
    all_doc_ids = set(bm25_ranks.keys()).union(set(dense_ranks.keys()))
    
    for doc_id in all_doc_ids:
        score = 0.0
        
        # Add BM25 component if it exists
        if doc_id in bm25_ranks:
            score += 1.0 / (k + bm25_ranks[doc_id])
            
        # Add Dense component if it exists
        if doc_id in dense_ranks:
            score += 1.0 / (k + dense_ranks[doc_id])
            
        rrf_scores[doc_id] = score
        
    # Sort by the final RRF score descending
    sorted_rrf = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_rrf

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Load the fully embedded data
    corpus_path_embeddings = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "sample_repository_output", "embeddings.json"))
    corpus_path_functions = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "sample_repository_output", "extracted_functions.json"))
    with open(corpus_path_embeddings, "r") as f:
        corpus_embeddings = json.load(f)
    with open(corpus_path_functions, "r") as f:
        corpus_functions = json.load(f)
        
    # Map documents by ID for easy lookup later
    doc_map = {doc["id"]: doc for doc in corpus_functions}

    # 2. Setup BM25 (Sparse)
    tokenized_corpus = [tokenize_code(doc["source_code"]) for doc in corpus_functions]
    bm25 = BM25Okapi(tokenized_corpus)

    # 3. The User Input
    original_query = "Which function handles cleaning text?"
    print(f"User Query: '{original_query}'\n")

    # 4. Expand Query (LLM)
    expanded_query = expand_query(original_query)
    
    # 5. Get Sparse Ranks (BM25) using expanded query
    tokenized_query = tokenize_code(expanded_query)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Pair doc_ids with scores, sort them, and map to ranks
    bm25_results = sorted(zip([doc["id"] for doc in corpus_functions], bm25_scores), key=lambda x: x[1], reverse=True)
    bm25_ranks = {doc_id: rank for rank, (doc_id, score) in enumerate(bm25_results, start=1)}

    # 6. Get Dense Ranks (Vector) using original query
    # (Dense retrieval doesn't usually need the LLM synonyms since it understands semantics inherently)
    dense_results = get_dense_rankings(original_query, corpus_embeddings)
    dense_ranks = {doc_id: rank for doc_id, score, rank in dense_results}

    # 7. Merge with Reciprocal Rank Fusion
    final_hybrid_ranking = calculate_rrf(bm25_ranks, dense_ranks)

    # 8. Display Final Results
    print("--- Final Hybrid Results (RRF) ---")
    for rank, (doc_id, rrf_score) in enumerate(final_hybrid_ranking[:3], start=1):
        function_id = doc_map[doc_id]["id"]
        print(f"Rank {rank} | RRF Score: {rrf_score:.4f} | Function: {function_id}")
        
        # Optional: Print the individual ranks to see how it was calculated
        print(f"          -> (BM25 Rank: {bm25_ranks.get(doc_id, 'N/A')}, Dense Rank: {dense_ranks.get(doc_id, 'N/A')})")