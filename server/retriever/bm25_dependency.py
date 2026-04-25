import networkx as nx
from retriever.hybrid_retrieval_dependency import get_neighborhood
from retriever.bm25_basic import bm25_search

def bm25_dependency(
    query: str,
    corpus: list[dict],
    G: nx.DiGraph,
    top_k: int = 3,
) -> list[tuple[str, str]]:

    bm25_ranking = bm25_search(query, corpus, top_k)
    initial_scores = {doc["id"]: score for doc, score in bm25_ranking}
    initial_function_ids = list(initial_scores.keys())

    result = initial_function_ids[:top_k]
    base = result  # default fallback

    if top_k == 3:
        base = initial_function_ids[:2]
        neighborhood = get_neighborhood(G, base)

        if len(neighborhood) > 0:
            result = base + list(neighborhood)[:1]
        else:
            result = initial_function_ids[:3]
            base = result  # fallback is pure BM25

    else:
        for i in range(2, -1, -1):
            num_retrieval_system = top_k - i
            base = initial_function_ids[:num_retrieval_system]

            neighborhood = get_neighborhood(G, base)
            result = base + list(neighborhood)[:i]

            if len(result) == top_k:
                break

    return [
        ("PRIMARY MATCH", doc_id)
        if doc_id in base
        else ("NEIGHBORING CONTEXT", doc_id)
        for doc_id in result
    ]
