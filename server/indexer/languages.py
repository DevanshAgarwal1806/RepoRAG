import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
import tree_sitter_java as tsjava
import tree_sitter_go as tsgo
import tree_sitter_rust as tsrust
import tree_sitter_cpp as tscpp
import tree_sitter_c as tsc
import tree_sitter_ruby as tsruby
import tree_sitter_php as tsphp
from tree_sitter import Language

QUERIES = {
    "python": """
        (function_definition
        name: (identifier) @name
        body: (block
            .
            (expression_statement
            (string
                (string_content) @docstring))?
            (_)*) @body)
    """
}

CALL_QUERIES = {
    "python": """
        (call function: (identifier) @callee)
        (call function: (attribute attribute: (identifier) @callee))
    """
}

IMPORT_QUERIES = {
    "python": """
        (import_statement (dotted_name) @imported_module)

        (import_statement
            (aliased_import
                name: (dotted_name) @imported_module
                alias: (identifier) @alias))

        (import_from_statement
            module_name: (dotted_name) @source_module
            (dotted_name) @imported_symbol)

        (import_from_statement
            module_name: (relative_import) @source_module
            (dotted_name) @imported_symbol)

        (import_from_statement
            module_name: (dotted_name) @source_module
            (aliased_import
                name: (dotted_name) @imported_symbol
                alias: (identifier) @alias))

        (import_from_statement
            module_name: (relative_import) @source_module
            (aliased_import
                name: (dotted_name) @imported_symbol
                alias: (identifier) @alias))

        (import_from_statement
            module_name: (dotted_name) @source_module
            (wildcard_import) @wildcard)

        (import_from_statement
            module_name: (relative_import) @source_module
            (wildcard_import) @wildcard)
    """
}

LANGUAGE_MAP = {
    "python": Language(tspython.language())
}

EXTENSION_MAP = {
    ".py": "python"
}
