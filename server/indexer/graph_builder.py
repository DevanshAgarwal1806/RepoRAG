import networkx as nx
from typing import List, Dict

from indexer.ast_parser import FunctionNode

def build_dependency_graph(
    functions: List[FunctionNode],
    global_import_map: Dict[str, dict] = None
) -> nx.DiGraph:
    
    G = nx.DiGraph()
    if global_import_map is None:
        global_import_map = {}

    exact_name_to_ids: Dict[str, List[str]] = {}
    base_name_to_ids: Dict[str, List[str]] = {}

    for fn in functions:
        G.add_node(fn.id, **{
            "name": fn.name,
            "language": fn.language,
            "file": fn.file_path,
            "start_line": fn.start_line,
            "end_line": fn.end_line,
            "docstring": fn.docstring,
            "source": fn.source_code,
            "is_method": fn.is_method
        })
        exact_name_to_ids.setdefault(fn.name, []).append(fn.id)
        if "." in fn.name:
            base_name = fn.name.split(".")[-1]
            base_name_to_ids.setdefault(base_name, []).append(fn.id)

    for fn in functions:
        file_imports = global_import_map.get(fn.file_path, {})
        for call_obj in fn.calls:
            called_name = call_obj["name"]
            is_method   = call_obj["is_method"]
            is_decorator = call_obj.get("is_decorator", False)
            receiver     = call_obj.get("receiver")
            if not is_method:
                if called_name in exact_name_to_ids:
                    for callee_id in exact_name_to_ids[called_name]:
                        callee_lang = G.nodes[callee_id].get("language")
                        weight = 1.0 if callee_lang == fn.language else 0.5
                        G.add_edge(fn.id, callee_id, weight=weight, is_decorator=is_decorator)

                elif called_name in base_name_to_ids:
                    for callee_id in base_name_to_ids[called_name]:
                        callee_lang = G.nodes[callee_id].get("language")
                        weight = 0.6 if callee_lang == fn.language else 0.3
                        G.add_edge(fn.id, callee_id, weight=weight, is_decorator=is_decorator)

                else:
                    lib_node_id = f"__external__::{called_name}"
                    if not G.has_node(lib_node_id):
                        G.add_node(lib_node_id, **{
                            "name": called_name,
                            "language": "external",
                            "file": "",
                            "start_line": 0,
                            "end_line": 0,
                            "docstring": None,
                            "source": "",
                            "embedding": None,
                            "is_method": False
                        })
                    G.add_edge(fn.id, lib_node_id, weight=0.1, is_decorator=is_decorator) 
            else:
                if called_name in base_name_to_ids:
                    for callee_id in base_name_to_ids[called_name]:
                        callee_node      = G.nodes[callee_id]
                        callee_full_name = callee_node["name"]
                        callee_file      = callee_node["file"]

                        # All class segments of the qualified name.
                        # e.g. "Outer.Inner.method" -> ["Outer", "Inner"]
                        # e.g. "MyClass.method"     -> ["MyClass"]
                        name_parts     = callee_full_name.split(".")
                        class_segments = name_parts[:-1]

                        # file_imports maps local_name -> module
                        # e.g. {"MyClass": "my_module", "np": "numpy", "B": "a"}
                        # We check if any class segment matches a known imported name (key)
                        # OR if any segment matches a module name (value) for cases like
                        # `import mymodule` then `mymodule.MyClass.method()`
                        imported_keys   = set(file_imports.keys())
                        imported_values = set(file_imports.values())

                        segment_set = set(class_segments)
                        
                        receiver_segments = set(receiver.split(".")) if receiver else set()
                        all_segments = segment_set | receiver_segments

                        if all_segments & imported_keys or all_segments & imported_values:
                            G.add_edge(fn.id, callee_id, weight=0.9, is_decorator=is_decorator)
                        elif callee_file == fn.file_path:
                            G.add_edge(fn.id, callee_id, weight=0.9, is_decorator=is_decorator)
                        else:
                            G.add_edge(fn.id, callee_id, weight=0.2, is_decorator=is_decorator)
                else:
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
                            "is_method":  False
                        })
                    G.add_edge(fn.id, lib_node_id, weight=0.1, is_decorator=is_decorator)

    return G