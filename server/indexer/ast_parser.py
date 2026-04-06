import os
import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from tree_sitter import Parser, Node, Tree, Language, Query, QueryCursor
from languages import LANGUAGE_MAP, EXTENSION_MAP, QUERIES, CALL_QUERIES, IMPORT_QUERIES

# Template for function nodes extracted from the AST
@dataclass
class FunctionNode:
    id: str # unique: "filepath::funcname::startline"
    name: str
    language: str
    file_path: str
    start_line: int
    end_line: int
    source_code: str
    docstring: Optional[str]
    calls: List[dict] = field(default_factory=list)
    embedding: Optional[list] = None

# Detects language based on file extension
def detect_language(file_path: str) -> Optional[str]:
    ext = os.path.splitext(file_path)[1].lower()
    return EXTENSION_MAP.get(ext)

# Extracts the full parent class name for a given function node, if it exists. This includes nested classes too.
def get_full_parent_class_name(node, source_bytes: bytes) -> Optional[str]:
    current = node.parent

    class_node_types = {
        "class_definition", "class_declaration", "struct_specifier",
        "impl_item", "type_declaration", "interface_declaration"
    }

    class_names: List[str] = []

    while current is not None:
        if current.type in class_node_types:
            # Find the class name inside this class node
            for child in current.children:
                if child.type in ("identifier", "type_identifier", "name"):
                    class_name = get_node_text(child, source_bytes)
                    class_names.append(class_name)
                    break  # stop after finding the name

        current = current.parent

    if not class_names:
        return None

    # Reverse because we collected from inner → outer
    return ".".join(reversed(class_names))

def get_node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

def extract_imports(
    tree: Tree,
    language: Language,
    language_name: str,
    source_bytes: bytes
) -> Dict[str, str]:

    if language_name not in IMPORT_QUERIES:
        return {}

    query = Query(language, IMPORT_QUERIES[language_name])
    cursor = QueryCursor(query)
    matches = cursor.matches(tree.root_node)

    import_map: Dict[str, str] = {}

    for _, captures in matches:

        # Normalize all capture lists
        def ensure_list(x):
            if x is None:
                return []
            return x if isinstance(x, list) else [x]

        module_nodes = ensure_list(captures.get("imported_module"))
        symbol_nodes = ensure_list(captures.get("imported_symbol"))
        alias_nodes = ensure_list(captures.get("alias"))
        source_nodes = ensure_list(captures.get("source_module"))
        wildcard_nodes = ensure_list(captures.get("wildcard"))

        # -----------------------------
        # Case 1: import x / import x as y
        # -----------------------------
        if module_nodes:
            # Build alias map using AST structure
            alias_map = {}

            for alias_node in alias_nodes:
                parent = alias_node.parent
                if parent:
                    for child in parent.children:
                        if child.type in ("dotted_name", "identifier"):
                            name = get_node_text(child, source_bytes)
                            alias_map[name] = get_node_text(alias_node, source_bytes)

            for module_node in module_nodes:
                module_name = get_node_text(module_node, source_bytes)

                if module_name in alias_map:
                    import_map[alias_map[module_name]] = module_name
                else:
                    import_map[module_name] = module_name

        # -----------------------------
        # Case 2: from x import *
        # -----------------------------
        elif source_nodes and wildcard_nodes:
            source_module = get_node_text(source_nodes[0], source_bytes)

            import_map.setdefault("__wildcard__", []).append(source_module)

        # -----------------------------
        # Case 3: from x import y / y as z
        # -----------------------------
        elif source_nodes and symbol_nodes:
            source_module = get_node_text(source_nodes[0], source_bytes)

            # Build alias map via AST
            alias_map = {}

            for alias_node in alias_nodes:
                parent = alias_node.parent
                if parent:
                    for child in parent.children:
                        if child.type in ("dotted_name", "identifier"):
                            symbol = get_node_text(child, source_bytes)
                            alias_map[symbol] = get_node_text(alias_node, source_bytes)

            for symbol_node in symbol_nodes:
                symbol_name = get_node_text(symbol_node, source_bytes)

                if symbol_name in alias_map:
                    import_map[alias_map[symbol_name]] = source_module
                else:
                    import_map[symbol_name] = source_module

    return import_map

def extract_calls(node, call_query, source_bytes, is_decorator: bool, calls_list, seen_calls):
    for _, caps in QueryCursor(call_query).matches(node):
        if "callee" in caps:
            callee_node = caps["callee"][0]
            call_name = get_node_text(callee_node, source_bytes)
            is_method = callee_node.parent.type in {
                "attribute", "member_expression", "property_identifier",
                "field_expression", "selector_expression", "qualified_identifier"
            }
            sig = (call_name, is_method, is_decorator)
            if sig not in seen_calls:
                seen_calls.add(sig)
                calls_list.append({
                    "name": call_name,
                    "is_method": is_method,
                    "is_decorator": is_decorator
                })

# Extracts functions from a single source file and returns a list of FunctionNodes along with the imports found in that file.
def extract_functions(file_path: str) -> tuple[List[FunctionNode], dict]:
    language_name = detect_language(file_path)
    if language_name not in LANGUAGE_MAP:
        return [], {}

    language = LANGUAGE_MAP[language_name]
    parser = Parser(language)

    with open(file_path, "rb") as f:
        source_bytes = f.read()

    tree = parser.parse(source_bytes)
    
    # Extract imports to build a import map for this file.
    file_import_map = extract_imports(tree, language, language_name, source_bytes)

    # Query to extract function and method definitions
    fn_query = Query(language, QUERIES[language_name])
    cursor = QueryCursor(fn_query)
    fn_matches = cursor.matches(tree.root_node)

    functions = []
    call_query = Query(language, CALL_QUERIES[language_name])
    for _, capture_dict in fn_matches:
        # Get required nodes from the captures
        name_node = capture_dict.get("name")
        body_node = capture_dict.get("body")
        doc_node = capture_dict.get("docstring")

        if not name_node or not body_node:
            continue

        # Get the function node (the parent of the body node).
        func_node = body_node[0].parent if isinstance(body_node, list) else body_node.parent
        if func_node is None:
            continue

        # If this function is decorated, expand the node to include the decorators
        if func_node.parent and func_node.parent.type == "decorated_definition":
            func_node = func_node.parent

        # Extract the name of the function/method.
        base_name_text = get_node_text(name_node[0] if isinstance(name_node, list) else name_node, source_bytes)
        # For methods, we want to prefix with the parent class name to create a unique identifier. For functions, this will just be the function name.
        parent_class = get_full_parent_class_name(func_node, source_bytes)
        name_text = f"{parent_class}.{base_name_text}" if parent_class else base_name_text
        
        # Source code (entire function)
        source_text = get_node_text(func_node, source_bytes)
        doc_text = get_node_text(doc_node[0] if isinstance(doc_node, list) else doc_node, source_bytes).strip('"\' \n') if doc_node else None

        calls_list = []
        seen_calls = set()

        if func_node.type == "decorated_definition":
            for deco_node in (c for c in func_node.children if c.type == "decorator"):
                extract_calls(deco_node, call_query, source_bytes, is_decorator=True, calls_list=calls_list, seen_calls=seen_calls)
            inner_func = next(
                c for c in func_node.children
                if c.type in ("function_definition")
            )
            extract_calls(inner_func.child_by_field_name("body"), call_query, source_bytes, is_decorator=False, calls_list=calls_list, seen_calls=seen_calls)
        else:
            extract_calls(func_node.child_by_field_name("body"), call_query, source_bytes, is_decorator=False, calls_list=calls_list, seen_calls=seen_calls)

        node_id = f"{file_path}::{name_text}::{func_node.start_point[0]}"

        functions.append(FunctionNode(
            id=node_id,
            name=name_text,
            language=language_name,
            file_path=file_path,
            start_line=func_node.start_point[0] + 1,
            end_line=func_node.end_point[0] + 1,
            source_code=source_text,
            docstring=doc_text,
            calls=calls_list,
        ))
        
    # -------------------------------------------------------------------------
    # Capture the __main__ guard block as a pseudo-function, if present.
    # We walk only the top-level children of the module so we never mistake a
    # nested if-statement (e.g. inside a function) for the entry-point block.
    # -------------------------------------------------------------------------
    if language_name == "python":
        for child in tree.root_node.children:
            if child.type != "if_statement":
                continue

            condition = child.child_by_field_name("condition")
            if condition is None:
                continue

            condition_text = get_node_text(condition, source_bytes)
            if "__name__" not in condition_text or "__main__" not in condition_text:
                continue

            body = child.child_by_field_name("consequence")
            if body is None:
                continue

            # Reuse the call extraction logic on this block
            calls_list = []
            seen_calls = set()
            extract_calls(child, call_query, source_bytes, is_decorator=False, calls_list=calls_list, seen_calls=seen_calls)

            node_id = f"{file_path}::__main__::{child.start_point[0]}"
            functions.append(FunctionNode(
                id=node_id,
                name="__main__",
                language=language_name,
                file_path=file_path,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                source_code=get_node_text(child, source_bytes),
                docstring=None,
                calls=calls_list,
            ))
            break  # there can only be one __main__ block per file

    return functions, file_import_map

def index_repository(repo_path: str) -> tuple[List[FunctionNode], dict]:
    """Walk the entire repo and extract functions from all supported files."""
    all_functions = []
    global_import_map = {}
    skipped = []

    for root, dirs, files in os.walk(repo_path):
        # Skip irrelevant directories
        dirs[:] = [d for d in dirs if d not in {
            ".git", "node_modules", "__pycache__", ".venv",
            "venv", "dist", "build", "target", ".idea", ".vscode"
        }]

        # Process each file in the current directory
        for file in files:
            full_path = os.path.join(root, file)
            lang = detect_language(full_path)
            if lang is None:
                continue
            try:
                fns, imports = extract_functions(full_path)
                all_functions.extend(fns)
                global_import_map[full_path] = imports
            except Exception as e:
                # Store the exact error message so we aren't guessing next time
                skipped.append((full_path, str(e)))

    print(f"Indexed {len(all_functions)} functions across "
          f"{len(set(fn.file_path for fn in all_functions))} files.")
    
    if skipped:
        print(f"Skipped {len(skipped)} files due to errors:")
        for skip in skipped:
            print(f" - {skip[0]}: {skip[1]}")

    return all_functions, global_import_map

def serialize_func_node(fn: FunctionNode) -> dict:
    return {
        "id": fn.id,
        "name": fn.name,
        "language": fn.language,
        "file_path": fn.file_path,
        "start_line": fn.start_line,
        "end_line": fn.end_line,
        "source_code": fn.source_code,
        "docstring": fn.docstring,
        "calls": fn.calls,
    }
    
def save_functions_to_json(functions: list[FunctionNode], output_file: str):
    data = [serialize_func_node(fn) for fn in functions]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    target_repo = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print(f"Starting parser on: {target_repo} ...")
    extracted_functions, extracted_imports = index_repository(target_repo)
    
    print("\n" + "="*60)
    print("GLOBAL IMPORT MAP (FILE -> {SYMBOL: MODULE})")
    print("="*60)
    print(json.dumps(extracted_imports, indent=4))
    
    output_json = "./extracted_functions.json"
    save_functions_to_json(extracted_functions, output_json)
    print(f"\nSaved extracted functions to {output_json}")