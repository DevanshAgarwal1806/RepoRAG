import networkx as nx
from retriever.hybrid_retrieval_dependency import propagate_scores_and_rerank
from retriever.bm25_basic import bm25_search

def bm25_dependency(
    query: str,
    corpus: list[dict],
    G: nx.DiGraph,
    top_k: int = 3,
) -> list[tuple[str, str]]:
    bm25_ranking = bm25_search(query, corpus, top_k)
    initial_scores = {doc["id"]: score for doc, score in bm25_ranking}
    corpus_ids = {doc["id"] for doc in corpus}
    function_ids = set(initial_scores)
    final_ranking = propagate_scores_and_rerank(
        G,
        initial_scores,
        top_k=top_k,
    )
    return [
        ("PRIMARY MATCH", doc_id)
        if doc_id in function_ids
        else ("NEIGHBORING CONTEXT", doc_id)
        for doc_id, _ in final_ranking
        if doc_id in corpus_ids
    ]
