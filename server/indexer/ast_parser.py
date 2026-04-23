import os
import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
from tree_sitter import Parser, Node, Tree, Language, Query, QueryCursor
from indexer.languages import LANGUAGE_MAP, EXTENSION_MAP, QUERIES, CALL_QUERIES, IMPORT_QUERIES

# Template for function nodes extracted from the AST
@dataclass
class FunctionNode:
    id: str # ID Format: "filepath[start_line:end_line]:funcname"
    name: str
    is_method: bool = False
    language: str
    file_path: str
    start_line: int
    end_line: int
    source_code: str
    docstring: Optional[str]
    calls: List[dict] = field(default_factory=list) # Initialize calls as an empty list by default, unique to each instance of the function node.

def get_language(file_path: str) -> Optional[str]:
    """
    Detects the language of the code based on the file extension

    Args:
        file_path (str): Path of the source file

    Returns:
        Optional[str]: Returns the name of the programming language. Returns None if the language is not supported or cannot be determined.
    """
    ext = os.path.splitext(file_path)[1].lower()
    return EXTENSION_MAP.get(ext)

def get_node_text(node: Node, source_code: bytes) -> str:
    """
    Returns the text of the node from the source code encoded in bytes.

    Args:
        node (Node): The node for which we want to extract the text.
        source_code (bytes): The source code encoded in bytes.

    Returns:
        str: The text of the node.
    """
    return source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

def get_full_parent_class_name(node: Node, source_code: bytes) -> Optional[str]:
    """
    Returns the complete parent class name for a function. It supports functions defined in nested classes. For multiple functions, they are returned in the format OuterClass.InnerClass. If the function is not defined within any class, it returns None.

    Args:
        node (Node): The node for which we want to extract the parent class name.
        source_code (bytes): The source code encoded in bytes.

    Returns:
        Optional[str]: The complete parent class name, or None if the function is not defined within any class.
    """
    
    current = node.parent

    class_node_types = {
        "class_definition",        # Python
        "class_declaration",       # Java, JS, TS
        "class_specifier",         # C++
        "struct_specifier",        # C/C++
        "interface_declaration",   # Java, TS
        "enum_declaration",        # Java, C/C++
        "impl_item",               # Rust
        "type_declaration",        # Go, others
    }

    class_names = []

    while current is not None:
        if current.type in class_node_types:
            for child in current.children:
                # For any node, the name is stored as a child node.
                if child.type in ("identifier", "type_identifier", "name"):
                    class_name = get_node_text(child, source_code)
                    class_names.append(class_name)
                    break

        current = current.parent

    if not class_names:
        return None

    return ".".join(reversed(class_names))

def ensure_list(x):
    """
    Converts x to a list if it is not already an instance of list.
    """
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def extract_imports(
    tree: Tree,
    language: Language,
    language_name: str,
    source_code: bytes
) -> Dict[str, str]:
    """
    Generates an import map, which maps imports to the modules from which it is imported.

    Args:
        tree (Tree): tree-sitter parse tree of the source file
        language (Language): Language object corresponding to the source file's programming language
        language_name (str): Name of the programming language
        source_code (bytes): Source code of the file encoded in bytes

    Returns:
        Dict[str, str]: A dictionary mapping imported symbols to their source modules.
    """

    # If language does not have a supported import query, return an empty map
    if language_name not in IMPORT_QUERIES:
        return {}

    # Extract query pattern for the language and create a query object
    query = Query(language, IMPORT_QUERIES[language_name])
    cursor = QueryCursor(query)
    matches = cursor.matches(tree.root_node)

    import_map = {}

    for _, captures in matches:
        module_nodes = ensure_list(captures.get("imported_module"))
        alias_nodes = ensure_list(captures.get("alias"))
        
        source_nodes = ensure_list(captures.get("source_module"))
        symbol_nodes = ensure_list(captures.get("imported_symbol"))
        
        wildcard_nodes = ensure_list(captures.get("wildcard"))

        # import x / import x as y
        # import module / import module as alias
        if module_nodes:
            alias_map = {}

            for alias_node in alias_nodes:
                parent = alias_node.parent
                if parent:
                    for child in parent.children:
                        if child.type in ("dotted_name", "identifier"):
                            name = get_node_text(child, source_code)
                            alias_map[name] = get_node_text(alias_node, source_code)

            for module_node in module_nodes:
                module_name = get_node_text(module_node, source_code)

                if module_name in alias_map:
                    import_map[alias_map[module_name]] = module_name
                else:
                    import_map[module_name] = module_name

        # from x import *
        # from source_module import *
        elif source_nodes and wildcard_nodes:
            source_module = get_node_text(source_nodes[0], source_code).strip("'\"")

            if "__wildcard__" in import_map:
                import_map["__wildcard__"].append(source_module)
            else:
                import_map["__wildcard__"] = [source_module]

        # Python:
        # from x import y / y as z
        # from source_module import symbol / symbol as alias
        # JS/TS:
        # import {symbol} from source_module
        # import symbol from source_module
        # import {symbol as alias} from source_module
        elif source_nodes and symbol_nodes:
            source_module = get_node_text(source_nodes[0], source_code).strip("'\"")

            alias_map = {}
            for alias_node in alias_nodes:
                parent = alias_node.parent
                if parent:
                    for child in parent.children:
                        if child.type in ("dotted_name", "identifier"):
                            symbol = get_node_text(child, source_code)
                            alias_map[symbol] = get_node_text(alias_node, source_code)

            for symbol_node in symbol_nodes:
                symbol_name = get_node_text(symbol_node, source_code)

                if symbol_name in alias_map:
                    import_map[alias_map[symbol_name]] = source_module
                else:
                    import_map[symbol_name] = source_module

    return import_map

def extract_calls(
    node: Node, 
    call_query: Query, 
    source_code: bytes,
    is_decorator: bool, 
    calls_list: List[dict], 
    seen_list: Set[str]
):
    """
    Extract function calls inside a function

    Args:
        node (Node): Function Node returned from the query matches for function definitions. This is the root node from which we will start looking for function calls.
        call_query (Query): Query object for extracting function calls
        source_code (bytes): Source code encoded in bytes
        is_decorator (bool): Indicates whether the current function is a decorator.
        calls_list (List[dict]): List to which the called function details are appended.
        seen_list (Set[str]): Set to keep track of already seen function calls to avoid duplicates.
    """
    if node is None:
        return

    for _, captures in QueryCursor(call_query).matches(node):
        if "callee" in captures:
            callee_capture = captures["callee"]
            callee_node = callee_capture[0] if isinstance(callee_capture, list) else callee_capture
            call_name = get_node_text(callee_node, source_code)
            if callee_node.parent.type in {
                "attribute", "member_expression", "property_identifier",
                "field_expression", "selector_expression", "qualified_identifier", "method_invocation"
            }:
                is_method = True
            else:
                is_method = False
                
            # Capture the receiver chain if present: a.b.someCall() -> "a.b"
            receiver_text = None
            if "receiver" in captures:
                receiver_node = captures["receiver"][0]
                receiver_text = get_node_text(receiver_node, source_code)
            
            signature = f"{call_name}|{is_method}|{is_decorator}|{receiver_text}"
            if signature in seen_list:
                continue
            seen_list.add(signature)
            calls_list.append({
                "name": call_name,
                "is_method": is_method,
                "is_decorator": is_decorator,
                "receiver": receiver_text
            })

def extract_functions(file_path: str) -> tuple[List[FunctionNode], dict]:
    """
    Extracts functions from the source file.

    Args:
        file_path (str): Path to the source file

    Returns:
        tuple[List[FunctionNode], dict]: Returns a tuple of extracted functions and an import map
    """
    language_name = get_language(file_path)
    if language_name not in LANGUAGE_MAP:
        return [], {}

    language = LANGUAGE_MAP[language_name]
    parser = Parser(language)

    with open(file_path, "rb") as f:
        source_bytes = f.read()

    # Generates the parse tree for the source file using tree-sitter and appropriate language determined using the file extension
    tree = parser.parse(source_bytes)
    
    # Extract imports to build a import map for this file.
    file_import_map = extract_imports(tree, language, language_name, source_bytes)

    # Query to extract function and method definitions
    fn_query = Query(language, QUERIES[language_name])
    cursor = QueryCursor(fn_query)
    fn_matches = cursor.matches(tree.root_node)
    
    seen_ids = set()

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
        if isinstance(body_node, list):
            func_node = body_node[0].parent
        else:
            func_node = body_node.parent
        
        if func_node is None:
            continue

        # Extracting complete function node for function expressions and arrow functions.
        if func_node.type in ("function_expression", "arrow_function"):
            if func_node.parent and func_node.parent.type == "variable_declarator":
                func_node = func_node.parent

        # Python: decorated functions
        if func_node.parent and func_node.parent.type == "decorated_definition":
            func_node = func_node.parent

        # Extract the name of the function/method.
        if name_node:
            if isinstance(name_node, list):
                name_node = name_node[0]
            base_name_text = get_node_text(name_node, source_bytes)
        else:
            base_name_text = "default_export"

        if language_name == "cpp" and "::" in base_name_text:
            base_name_text = base_name_text.replace("::", ".")
            
        # For methods, we want to prefix with the parent class name to create a unique identifier. For functions, this will just be the function name.
        parent_class = get_full_parent_class_name(func_node, source_bytes)
        name_text = f"{parent_class}.{base_name_text}" if parent_class else base_name_text
        
        # Source code (entire function)
        source_node = func_node
        if func_node.type == "variable_declarator" and func_node.parent and func_node.parent.type in ("lexical_declaration", "variable_declaration"):
            source_node = func_node.parent
        if source_node.parent and source_node.parent.type == 'export_statement':
            source_node = source_node.parent
            
        # Extract code for the function
        source_text = get_node_text(source_node, source_bytes)
        
        if isinstance(doc_node, list):
            doc_node = doc_node[0]
        if doc_node:
            doc_text = get_node_text(doc_node, source_bytes).strip('"\' \n')
        else:
            doc_text = None

        calls_list = list()
        seen_calls = set()

        FUNC_DEF_TYPES = {
            "function_definition",
            "function_declaration",
            "function_expression",
            "arrow_function",
            "method_definition"
        }

        if func_node.type == "decorated_definition":
            decorator_nodes = [c for c in func_node.children if c.type == "decorator"]
            for deco_node in decorator_nodes:
                extract_calls(deco_node, call_query, source_bytes, is_decorator=True, calls_list=calls_list, seen_list=seen_calls)
                
            inner_func = next(c for c in func_node.children if c.type in FUNC_DEF_TYPES)
            extract_calls(inner_func.child_by_field_name("body"), call_query, source_bytes, is_decorator=False, calls_list=calls_list, seen_list=seen_calls)
            
        elif func_node.type == "variable_declarator":
            value = func_node.child_by_field_name("value")
            if value:
                body = value.child_by_field_name("body")
                if body:
                    extract_calls(body, call_query, source_bytes, is_decorator=False, calls_list=calls_list, seen_list=seen_calls)
                    
        else:
            body = func_node.child_by_field_name("body")
            if body:
                extract_calls(body, call_query, source_bytes, is_decorator=False, calls_list=calls_list, seen_list=seen_calls)

        node_id = f"{file_path}[{func_node.start_point[0] + 1}-{func_node.end_point[0] + 1}]:{name_text}"
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)

        functions.append(FunctionNode(
            id=node_id,
            name=name_text,
            is_method="." in name_text,
            language=language_name,
            file_path=file_path,
            start_line=func_node.start_point[0] + 1,
            end_line=func_node.end_point[0] + 1,
            source_code=source_text,
            docstring=doc_text,
            calls=calls_list,
        ))
        
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

            calls_list = []
            seen_calls = set()
            extract_calls(child, call_query, source_bytes, is_decorator=False, calls_list=calls_list, seen_list=seen_calls)

            node_id = f"{file_path}[{child.start_point[0] + 1}-{child.end_point[0] + 1}]:__main__"
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
                is_method=False
            ))
            break

    return functions, file_import_map

def index_repository(repo_path: str) -> tuple[List[FunctionNode], dict]:
    """
    Indexes the complete repository by extracting the functions from each source file in the repository.

    Args:
        repo_path (str): Path to the repository

    Returns:
        tuple[List[FunctionNode], dict]: Returns a list of functions extracted and imports map
    """
    
    all_functions = []
    global_import_map = {}
    skipped = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {
            ".git", "node_modules", "__pycache__", ".venv",
            "venv", "dist", "build", "target", ".idea", ".vscode"
        }]

        # Process each file in the current directory
        for file in files:
            full_path = os.path.join(root, file)
            lang = get_language(full_path)
            if lang is None:
                continue
            try:
                fns, imports = extract_functions(full_path)
                all_functions.extend(fns)
                global_import_map[full_path] = imports
            except Exception as e:
                skipped.append((full_path, str(e)))

    print(f"Indexed {len(all_functions)} functions across {len(set(fn.file_path for fn in all_functions))} files.")
    
    if skipped:
        print(f"Skipped {len(skipped)} files due to errors:")
        for skip in skipped:
            print(f" - {skip[0]}: {skip[1]}")

    return all_functions, global_import_map

def serialize_func_node(fn: FunctionNode) -> dict:
    """
    Converts the function node into a dictionary for serializing.

    Args:
        fn (FunctionNode): Function Node to be serialized

    Returns:
        dict: Dictionary representation of the function node
    """
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
        "is_method": fn.is_method
    }
    
def save_functions_to_json(functions: list[FunctionNode], output_file: str) -> None:
    """
    Save functions extracted to a json file.

    Args:
        functions (list[FunctionNode]): List of function nodes to be saved
        output_file (str): Path to the output JSON file
    """
    data = [serialize_func_node(fn) for fn in functions]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    target_repo = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print(f"Starting parser on: {target_repo} ...")
    extracted_functions, extracted_imports = index_repository(target_repo)
    
    print("Import Map structured as {SYMBOL: MODULE}:")
    print(json.dumps(extracted_imports, indent=4))
    
    output_json = "./extracted_functions.json"
    save_functions_to_json(extracted_functions, output_json)
    print(f"\nSaved extracted functions to {output_json}")
