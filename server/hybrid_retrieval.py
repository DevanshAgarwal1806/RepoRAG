import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import json
from pathlib import Path
from rank_bm25 import BM25Okapi
import argparse

from retriever.query_expansion import expand_query
from retriever.bm25_basic import tokenize_code # Reusing your regex tokenizer
from retriever.dense_retrieval import get_dense_rankings

from retriever.graph_context import assemble_llm_context

def calculate_rrf(bm25_ranks: dict, dense_ranks: dict, k: int = 60) -> list[tuple[str, float]]:
    """
    Combines ranks using Reciprocal Rank Fusion.
    Formula: RRF_Score = sum( 1 / (k + rank) )
    """
    rrf_scores = {}
    all_doc_ids = set(bm25_ranks.keys()).union(set(dense_ranks.keys()))
    
    for doc_id in all_doc_ids:
        score = 0.0
        if doc_id in bm25_ranks:
            score += 1.0 / (k + bm25_ranks[doc_id])
        if doc_id in dense_ranks:
            score += 1.0 / (k + dense_ranks[doc_id])
        rrf_scores[doc_id] = score
        
    sorted_rrf = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_rrf

def run_hybrid_retrieval(output_dir: Path, query: str):
    # 1. Load the fully embedded data
    corpus_path_embeddings = output_dir / "embeddings.json"
    corpus_path_functions = output_dir / "extracted_functions.json"
    
    with open(corpus_path_embeddings, "r", encoding="utf-8") as f:
        corpus_embeddings = json.load(f)
    with open(corpus_path_functions, "r", encoding="utf-8") as f:
        corpus_functions = json.load(f)
        
    doc_map = {doc["id"]: doc for doc in corpus_functions}

    # 2. Setup BM25 (Sparse)
    tokenized_corpus = [tokenize_code(doc["source_code"]) for doc in corpus_functions]
    bm25 = BM25Okapi(tokenized_corpus)

    # 3. The User Input
    original_query = query
    print(f"User Query: '{original_query}'")

    # 4. Expand Query (LLM)
    expanded_query = expand_query(original_query)
    
    # 5. Get Sparse Ranks (BM25)
    tokenized_query = tokenize_code(expanded_query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_results = sorted(zip([doc["id"] for doc in corpus_functions], bm25_scores), key=lambda x: x[1], reverse=True)
    bm25_ranks = {doc_id: rank for rank, (doc_id, score) in enumerate(bm25_results, start=1)}

    # 6. Get Dense Ranks (Vector)
    dense_results = get_dense_rankings(original_query, corpus_embeddings)
    dense_ranks = {doc_id: rank for doc_id, score, rank in dense_results}

    # 7. Merge with Reciprocal Rank Fusion
    final_hybrid_ranking = calculate_rrf(bm25_ranks, dense_ranks)

    # 8. Extract the REAL Top K Results
    K_LIMIT = 10
    top_k_ids = []
    
    print("\n--- Final Hybrid Results (RRF) ---")
    for rank, (doc_id, rrf_score) in enumerate(final_hybrid_ranking[:K_LIMIT], start=1):
        function_id = doc_map[doc_id]["id"]
        top_k_ids.append(function_id)  # Collect the ID for the graph builder
        print(f"Rank {rank} | BM25 Rank: {bm25_ranks.get(doc_id, 'N/A')} | Dense Rank: {dense_ranks.get(doc_id, 'N/A')} | RRF Score: {rrf_score:.4f} | Function: {function_id}")

    # 9. Build the LLM Context Package
    print("\n--- Generating Graph Context ---")
    # We pass the real IDs we just found, and set depth (d) to 1
    final_prompt, context_nodes = assemble_llm_context(top_k_ids, str(output_dir), d=1)
    
    if len(context_nodes) > 0:
        print(f"\nNeighbouring Context: ")
        for node_id in context_nodes:
            print(f"Function: {node_id}")
    
    # 10. Save it to a markdown file for the LLM (and so you can read it easily)
    prompt_file = output_dir / "final_llm_payload.md"
    with open(prompt_file, "w", encoding="utf-8") as f:
        # We append the original query to the top so the LLM knows what to answer!
        f.write(f"### USER QUERY: {original_query}\n\n")
        f.write(final_prompt)
        
    print(f"Successfully generated full RAG payload: {len(final_prompt)} characters.")
    print(f"Saved to: {prompt_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Retrieval with RRF and Graph Context")
    parser.add_argument("--output", "-o", required=True, help="Output directory for results")
    parser.add_argument("--query", "-q", required=True, help="User query to search for")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    query = args.query

    run_hybrid_retrieval(output_dir, query)