import os
import json
import networkx as nx
from collections import deque
from pathlib import Path

def load_data(output_dir: str):
    """Loads the dependency graph and the extracted functions."""
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

def get_neighborhood(G: nx.DiGraph, start_nodes: list[str], max_depth: int = 1, min_weight: float = 0.4) -> set[str]:
    """
    Performs a bidirectional BFS to find neighbors up to max_depth.
    Only traverses edges that meet the min_weight threshold.
    """
    context_nodes = set(start_nodes)
    
    for start_node in start_nodes:
        if start_node not in G:
            continue
            
        # Queue stores: (node_id, current_depth)
        queue = deque([(start_node, 0)])
        visited = {start_node}
        
        while queue:
            current_node, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
                
            # 1. Look at Children (functions the current_node calls)
            for child in G.successors(current_node):
                edge_data = G.get_edge_data(current_node, child)
                if edge_data.get("weight", 0) >= min_weight and child not in visited:
                    visited.add(child)
                    context_nodes.add(child)
                    queue.append((child, depth + 1))
                    
            # 2. Look at Parents (functions that call the current_node)
            for parent in G.predecessors(current_node):
                edge_data = G.get_edge_data(parent, current_node)
                if edge_data.get("weight", 0) >= min_weight and parent not in visited:
                    visited.add(parent)
                    context_nodes.add(parent)
                    queue.append((parent, depth + 1))
                    
    return context_nodes

def assemble_llm_context(top_k_ids: list[str], output_dir: str, d: int = 1) -> str:
    """
    Main function to wrap everything up. Returns a formatted string ready for the LLM.
    """
    G, function_map = load_data(output_dir)
    
    # Get all relevant node IDs (the core k + their neighbors)
    all_context_ids = get_neighborhood(G, top_k_ids, max_depth=d)
    
    # Format the payload
    llm_prompt_context = "### CODEBASE CONTEXT ###\n\n"
    
    for node_id in all_context_ids:
        # Filter out external/library nodes (they don't have source code in our map)
        if node_id.startswith("__external__") or node_id not in function_map:
            continue
            
        fn_data = function_map[node_id]
        role = "PRIMARY MATCH" if node_id in top_k_ids else "NEIGHBORING CONTEXT"
        
        # Safely extract the file path and source code
        file_path = fn_data.get("file_path", fn_data.get("file", "Unknown File"))
        source_code = fn_data.get("source_code", fn_data.get("source", "No source available."))
        
        llm_prompt_context += f"--- {role}: {fn_data.get('name', 'Unknown')} ---\n"
        llm_prompt_context += f"File: {file_path}\n"
        llm_prompt_context += f"Code:\n```\n{source_code}\n```\n\n"
        
    return llm_prompt_context


if __name__ == "__main__":
    
    
    # This securely resolves the absolute path regardless of where you run the script from
    current_file_path = Path(__file__).resolve()
    
    # .parent goes to 'retriever', .parent again goes to 'server'
    server_dir = current_file_path.parent.parent
    output_dir = server_dir / "sample_repository_output"
    
    print(f"Looking for data in: {output_dir}") # Debug print
    
    mock_top_k = [
        "sample_repository/text_processor.py::DocumentProcessor.clean_text::15",
        "sample_repository/utils.ts::calculate_bm25::10",
        "sample_repository/math_utils.py::normalize_vector::5"
    ]
    
    # Build the context with d=1
    final_prompt_string = assemble_llm_context(mock_top_k, str(output_dir), d=1)
    
    print("\nSuccessfully built LLM context payload!")
    print(f"Payload length: {len(final_prompt_string)} characters.")
    
    # Save the string to a markdown file to inspect it easily
    prompt_file = output_dir / "test_prompt_context.md"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(final_prompt_string)
        
    print(f"Saved prompt text to: {prompt_file}")