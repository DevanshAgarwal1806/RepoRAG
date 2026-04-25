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

def hybrid_retrieval(query: str, corpus: list[dict], embeddings: dict, top_k: int = 3):
    # 2. Setup BM25 (Sparse)
    tokenized_corpus = [tokenize_code(doc["source_code"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # 5. Get Sparse Ranks (BM25)
    tokenized_query = tokenize_code(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_results = sorted(zip([doc["id"] for doc in corpus], bm25_scores), key=lambda x: x[1], reverse=True)
    bm25_ranks = {doc_id: rank for rank, (doc_id, score) in enumerate(bm25_results, start=1)}

    # 6. Get Dense Ranks (Vector)
    dense_results = get_dense_rankings(query, embeddings)
    dense_ranks = {doc_id: rank for doc_id, score, rank in dense_results}

    # 7. Merge with Reciprocal Rank Fusion
    final_hybrid_ranking = calculate_rrf(bm25_ranks, dense_ranks)
    return final_hybrid_ranking[:top_k]
