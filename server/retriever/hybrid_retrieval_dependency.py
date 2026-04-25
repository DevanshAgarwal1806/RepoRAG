import os
import json
import networkx as nx
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

from retriever.hybrid_retrieval import hybrid_retrieval

def load_data(output_dir: str):
    """
    Loads the dependency graph and the extracted functions.
    """
    graph_path = os.path.join(output_dir, "dependency_graph.json")
    functions_path = os.path.join(output_dir, "extracted_functions.json")
    
    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
        
        # Determine if the JSON uses 'links' or 'edges' to prevent KeyErrors
        edge_key = "links" if "links" in graph_data else "edges"
        
        # Pass the explicit key to the graph builder
        G = nx.node_link_graph(graph_data, edges=edge_key)
        
    with open(functions_path, "r", encoding="utf-8") as f:
        functions = json.load(f)
        # Map IDs to their full data for O(1) lookup
        function_map = {fn["id"]: fn for fn in functions}
        
    return G, function_map

def get_neighborhood(
    G: nx.DiGraph,
    start_node_ids: list[str],
    max_depth: int = 1,
    min_weight: float = 0.4
) -> list[str]:

    start_nodes = {n for n in start_node_ids if n in G}
    context_nodes = set(start_nodes)
    context_nodes_list = []

    for start_node in start_nodes:
        queue = deque([(start_node, 0)])
        visited = {start_node}

        while queue:
            current_node, depth = queue.popleft()

            if depth >= max_depth:
                continue

            # Children
            for child in G.successors(current_node):
                edge_data = G.get_edge_data(current_node, child)
                weight = edge_data.get("weight", 0) if edge_data else 0

                if weight >= min_weight and child not in visited:
                    visited.add(child)

                    if child not in context_nodes:
                        context_nodes.add(child)
                        context_nodes_list.append((child, weight))

                    queue.append((child, depth + 1))

            # Parents
            for parent in G.predecessors(current_node):
                edge_data = G.get_edge_data(parent, current_node)
                weight = edge_data.get("weight", 0) if edge_data else 0

                if weight >= min_weight and parent not in visited:
                    visited.add(parent)

                    if parent not in context_nodes:
                        context_nodes.add(parent)
                        context_nodes_list.append((parent, weight))

                    queue.append((parent, depth + 1))

    # Sort by weight descending
    context_nodes_list.sort(key=lambda x: x[1], reverse=True)

    return [node_id for node_id, _ in context_nodes_list]

def propagate_scores_and_rerank(
    G: nx.DiGraph, 
    initial_search_results: Dict[str, float], 
    alpha: float = 0.85, 
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Takes initial semantic/BM25 search scores and propagates them through 
    the AST Dependency Graph to boost the ranking of critical dependencies.
    
    Args:
        G: The NetworkX directional graph representing your codebase AST.
        initial_search_results: A dictionary of {node_id: search_score} from your vector DB.
        alpha: Damping factor. Higher = spreads further. Lower = stays closer to original hits.
        top_k: How many final nodes to return.
    """
    
    # 1. Initialize the "Heat" Map
    # Every node in the graph starts cold (0.0)
    personalization = {node: 0.0 for node in G.nodes()}
    
    # 2. Inject the Initial Search Scores (The "Heat Sources")
    valid_initial_nodes = [node_id for node_id in initial_search_results if node_id in personalization]
    has_positive_scores = any(initial_search_results[node_id] > 0 for node_id in valid_initial_nodes)

    for rank, node_id in enumerate(valid_initial_nodes, start=1):
        score = initial_search_results[node_id]
        if node_id in personalization:
            if has_positive_scores:
                personalization[node_id] = max(0.0, score) # Ensure no negative BM25 scores
            else:
                personalization[node_id] = 1.0 / rank
            
    # Safety Check: If the search engine found absolutely nothing in the graph
    if sum(personalization.values()) == 0:
        print("Warning: Initial search yielded no valid graph nodes.")
        return []

    # 3. Run the Propagation (Spreading Activation)
    # NetworkX handles the iterative matrix math automatically until convergence.
    # It will use the 'weight' attribute on your edges if you assigned one during parsing.
    propagated_scores = nx.pagerank(
        G, 
        alpha=alpha, 
        personalization=personalization, 
        weight='weight' 
    )
    
    # 4. Sort all nodes by their new, network-aware scores
    ranked_nodes = sorted(
        propagated_scores.items(), 
        key=lambda item: item[1], 
        reverse=True
    )
    
    # 5. Return the Top K Context for the LLM
    return ranked_nodes[:top_k]

def hybrid_retrieval_with_dependency(
    query: str,
    corpus: list[dict],
    embeddings: list[dict],
    G: nx.DiGraph,
    top_k: int = 3,
) -> list[tuple[str, str]]:
    final_hybrid_ranking = hybrid_retrieval(query, corpus, embeddings, top_k)
    corpus_ids = {doc["id"] for doc in corpus}
    function_ids = [doc_id for doc_id, score in final_hybrid_ranking]
    final_ranking = propagate_scores_and_rerank(
        G,
        {doc_id: rank for doc_id, rank in final_hybrid_ranking},
        top_k=top_k,
    )
    return [
        ("PRIMARY MATCH", doc_id)
        if doc_id in function_ids
        else ("NEIGHBORING CONTEXT", doc_id)
        for doc_id, _ in final_ranking
        if doc_id in corpus_ids
    ]

def hybrid_retrieval_with_dependency_1(
    query: str,
    corpus: list[dict],
    embeddings: list[dict],
    G: nx.DiGraph,
    top_k: int = 3,
) -> list[tuple[str, str]]:
    initial_hybrid_ranking = hybrid_retrieval(query, corpus, embeddings, top_k=top_k)
    initial_scores = {doc_id: score for doc_id, score in initial_hybrid_ranking}
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
