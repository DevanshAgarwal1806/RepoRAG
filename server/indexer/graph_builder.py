import networkx as nx
from typing import List, Dict

from ast_parser import FunctionNode

def build_dependency_graph(
    functions: List[FunctionNode],
    global_import_map: Dict[str, dict] = None
) -> nx.DiGraph:
    """
    Builds a directed graph of function dependencies using exact and fuzzy heuristic matching.

    :param functions: List of extracted FunctionNodes from the AST parser.
    :param global_import_map: Dictionary mapping file paths to their localized imports.
    :return: A NetworkX DiGraph representing the codebase structure.
    """
    G = nx.DiGraph()
    if global_import_map is None:
        global_import_map = {}

    # ── Index Phase ────────────────────────────────────────────────────────────
    # exact_name_to_ids: full qualified name -> [node_ids]
    #   e.g. "DocumentProcessor.clean_text" -> ["path/processor.py::DocumentProcessor.clean_text::45"]
    #   e.g. "calculate_bm25"               -> ["path/utils.py::calculate_bm25::10"]
    #
    # base_name_to_ids: method base name only -> [node_ids], populated only for methods
    #   e.g. "clean_text" -> ["path/processor.py::DocumentProcessor.clean_text::45"]
    #   Used when a call site writes processor.clean_text() — the parser only sees "clean_text"

    exact_name_to_ids: Dict[str, List[str]] = {}
    base_name_to_ids: Dict[str, List[str]] = {}

    for fn in functions:
        G.add_node(fn.id, **{
            "name":       fn.name,
            "language":   fn.language,
            "file":       fn.file_path,
            "start_line": fn.start_line,
            "end_line":   fn.end_line,
            "docstring":  fn.docstring,
            "source":     fn.source_code
        })

        exact_name_to_ids.setdefault(fn.name, []).append(fn.id)

        if "." in fn.name:
            base_name = fn.name.split(".")[-1]
            base_name_to_ids.setdefault(base_name, []).append(fn.id)

    # ── Edge Resolution Phase ──────────────────────────────────────────────────
    for fn in functions:
        file_imports = global_import_map.get(fn.file_path, {})

        for call_obj in fn.calls:
            called_name = call_obj["name"]
            is_method   = call_obj["is_method"]
            is_decorator = call_obj.get("is_decorator", False)

            # ── Case 1: Standalone calls e.g. calculate_bm25() ────────────────
            if not is_method:
                if called_name in exact_name_to_ids:
                    # High confidence: exact match on a module-level function name
                    for callee_id in exact_name_to_ids[called_name]:
                        callee_lang = G.nodes[callee_id].get("language")
                        weight = 1.0 if callee_lang == fn.language else 0.5
                        G.add_edge(fn.id, callee_id, weight=weight, is_decorator=is_decorator)

                elif called_name in base_name_to_ids:
                    # Medium confidence: name matches a method base name — could be a
                    # standalone call to a function that happens to share the method name,
                    # or a direct call without the object prefix (e.g. inside the class itself)
                    for callee_id in base_name_to_ids[called_name]:
                        callee_lang = G.nodes[callee_id].get("language")
                        weight = 0.6 if callee_lang == fn.language else 0.3
                        G.add_edge(fn.id, callee_id, weight=weight, is_decorator=is_decorator)

                else:
                    # No internal match — treat as an external/library call
                    lib_node_id = f"__external__::{called_name}"
                    if not G.has_node(lib_node_id):
                        G.add_node(lib_node_id, **{
                            "name":       called_name,
                            "language":   "external",
                            "file":       "",
                            "start_line": 0,
                            "end_line":   0,
                            "docstring":  None,
                            "source":     "",
                            "embedding":  None,
                        })
                    G.add_edge(fn.id, lib_node_id, weight=0.1, is_decorator=is_decorator)

            # ── Case 2: Method calls e.g. processor.clean_text() ──────────────
            else:
                if called_name in base_name_to_ids:
                    for callee_id in base_name_to_ids[called_name]:
                        callee_node      = G.nodes[callee_id]
                        callee_full_name = callee_node["name"]
                        callee_file      = callee_node["file"]

                        # The class name is the first segment of the qualified name
                        # e.g. "DocumentProcessor.clean_text" -> "DocumentProcessor"
                        class_name = callee_full_name.split(".")[0]

                        # High confidence: class is explicitly imported in the caller's file
                        # file_imports maps symbol -> module, so class names are keys
                        # e.g. {"DocumentProcessor": "document_processor", ...}
                        if class_name in file_imports:
                            G.add_edge(fn.id, callee_id, weight=0.9, is_decorator=is_decorator)

                        # High confidence: class is defined in the same file as the caller
                        elif callee_file == fn.file_path:
                            G.add_edge(fn.id, callee_id, weight=0.9, is_decorator=is_decorator)

                        # Low confidence: method name matches but no import evidence —
                        # noisy edge kept intentionally as an IR fallback since a missed
                        # link is worse than a low-weight false positive for graph traversal
                        else:
                            G.add_edge(fn.id, callee_id, weight=0.2, is_decorator=is_decorator)

                else:
                    # No internal match — external method call e.g. requests.get()
                    lib_node_id = f"__external__::{called_name}"
                    if not G.has_node(lib_node_id):
                        G.add_node(lib_node_id, **{
                            "name":       called_name,
                            "language":   "external",
                            "file":       "",
                            "start_line": 0,
                            "end_line":   0,
                            "docstring":  None,
                            "source":     "",
                            "embedding":  None,
                        })
                    G.add_edge(fn.id, lib_node_id, weight=0.1, is_decorator=is_decorator)

    return G