import networkx as nx
from typing import List, Dict

# Assumes this is placed in the same directory as ast_parser.py
from .ast_parser import FunctionNode

def build_dependency_graph(functions: List[FunctionNode], global_import_map: Dict[str, dict] = None) -> nx.DiGraph:
    """
    Builds a directed graph of function dependencies using exact and fuzzy heuristic matching.
    
    :param functions: List of extracted FunctionNodes from the AST parser.
    :param global_import_map: Dictionary mapping file paths to their localized imports.
    :return: A NetworkX DiGraph representing the codebase structure.
    """
    G = nx.DiGraph()
    if global_import_map is None:
        global_import_map = {}

    # We maintain two indices for smart resolution
    exact_name_to_ids = {}
    base_name_to_ids = {}

    for fn in functions:
        # Add node (fn.name already contains the ClassName if it's a method)
        G.add_node(fn.id, **{
            "name":       fn.name,
            "language":   fn.language,
            "file":       fn.file_path,
            "start_line": fn.start_line,
            "end_line":   fn.end_line,
            "docstring":  fn.docstring,
            "source":     fn.source_code,
            "embedding":  fn.embedding,
        })
        
        # Index 1: Exact match (e.g., 'DocumentProcessor.clean_text' or 'calculate_bm25')
        exact_name_to_ids.setdefault(fn.name, []).append(fn.id)
        
        # Index 2: Base method match (e.g., 'clean_text')
        if "." in fn.name:
            base_name = fn.name.split(".")[-1]
            base_name_to_ids.setdefault(base_name, []).append(fn.id)

    # ── Edge Resolution Phase ──────────────────────────────────────────────────
    for fn in functions:
        # Fetch the imports available in the file where this function lives
        file_imports = global_import_map.get(fn.file_path, {})

        for call_obj in fn.calls:
            called_name = call_obj["name"]
            is_method = call_obj["is_method"]

            # 1. STANDALONE CALLS (e.g., calculate_bm25())
            if not is_method:
                # Only look for exact matches. Avoids falsely linking to DocumentProcessor.calculate_bm25
                if called_name in exact_name_to_ids:
                    for callee_id in exact_name_to_ids[called_name]:
                        callee_lang = G.nodes[callee_id].get("language")
                        # Reward intra-language calls with higher weight
                        weight = 1.0 if callee_lang == fn.language else 0.5
                        G.add_edge(fn.id, callee_id, weight=weight)
            
            # 2. OBJECT METHOD CALLS (e.g., processor.clean_text())
            elif is_method:
                # Trigger the heuristic class resolution
                if called_name in base_name_to_ids:
                    for callee_id in base_name_to_ids[called_name]:
                        callee_node = G.nodes[callee_id]
                        callee_full_name = callee_node["name"]
                        callee_file = callee_node["file"]
                        
                        # Extract the class name of the candidate ('DocumentProcessor')
                        class_name = callee_full_name.split(".")[0]
                        
                        # ── Contextual Filtering ──
                        
                        # High Confidence: The class is imported in the caller's file
                        if class_name in file_imports.values() or class_name in file_imports.keys():
                            G.add_edge(fn.id, callee_id, weight=0.9)
                        
                        # High Confidence: The class is defined in the EXACT same file as the caller
                        elif callee_file == fn.file_path:
                            G.add_edge(fn.id, callee_id, weight=0.9)
                        
                        # Low Confidence: Fuzzy match (IR fallback for dynamic typing)
                        # The class wasn't imported, but the method name matches. We add it with a 
                        # low weight. In Information Retrieval, having a slightly noisy edge is often 
                        # better than missing a crucial link entirely.
                        else:
                            G.add_edge(fn.id, callee_id, weight=0.2)

    return G