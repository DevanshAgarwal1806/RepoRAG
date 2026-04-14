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
    """,
    "javascript": """
        (function_declaration
            name: (identifier) @name
            body: (statement_block) @body)

        (method_definition
            name: (property_identifier) @name
            body: (statement_block) @body)

        (variable_declarator
            name: (identifier) @name
            value: (function_expression
                body: (statement_block) @body))

        (variable_declarator
            name: (identifier) @name
            value: (arrow_function
                body: (statement_block) @body))
    """,
    "typescript": """
        (function_declaration
            name: (identifier) @name
            body: (statement_block) @body)

        (method_definition
            name: (property_identifier) @name
            body: (statement_block) @body)

        (variable_declarator
            name: (identifier) @name
            value: (function_expression
                body: (statement_block) @body))

        (variable_declarator
            name: (identifier) @name
            value: (arrow_function
                body: (statement_block) @body))
                
        (export_statement
            declaration: (lexical_declaration
                (variable_declarator
                    name: (identifier) @name
                    value: (arrow_function
                        body: [(statement_block) (expression)] @body))))
                        
        (export_statement
            value: (arrow_function
                body: [(statement_block) (expression)] @body) @is_default_export)
    """,
    "tsx": """
        (function_declaration
            name: (identifier) @name
            body: (statement_block) @body)

        (method_definition
            name: (property_identifier) @name
            body: (statement_block) @body)

        (variable_declarator
            name: (identifier) @name
            value: (function_expression
                body: (statement_block) @body))

        (variable_declarator
            name: (identifier) @name
            value: (arrow_function
                body: (statement_block) @body))
        
        (export_statement
            declaration: (lexical_declaration
                (variable_declarator
                    name: (identifier) @name
                    value: (arrow_function
                        body: [(statement_block) (expression)] @body))))

        (export_statement
            value: (arrow_function
                body: [(statement_block) (expression)] @body) @is_default_export)
    """,
}

CALL_QUERIES = {
    "python": """
        (call function: (identifier) @callee)
        (call function: (attribute attribute: (identifier) @callee))
    """,
    "javascript": """
        (call_expression function: (identifier) @callee)
        (call_expression function: (member_expression property: (property_identifier) @callee))
    """,
    "typescript": """
        (call_expression function: (identifier) @callee)
        (call_expression function: (member_expression property: (property_identifier) @callee))
    """,
    "tsx": """
        (call_expression function: (identifier) @callee)
        (call_expression function: (member_expression property: (property_identifier) @callee))
    """,

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
    """,
    "javascript": """
        (import_statement
            (import_clause (identifier) @imported_symbol)
            source: (string) @source_module)

        (import_statement
            (import_clause (named_imports
                (import_specifier name: (identifier) @imported_symbol)))
            source: (string) @source_module)

        (import_statement
            (import_clause (named_imports
                (import_specifier
                    name: (identifier) @imported_symbol
                    alias: (identifier) @alias)))
            source: (string) @source_module)

        (import_statement
            (import_clause (namespace_import (identifier) @imported_symbol))
            source: (string) @source_module)
    """,
    "typescript": """
        (import_statement
            (import_clause (identifier) @imported_symbol)
            source: (string) @source_module)

        (import_statement
            (import_clause (named_imports
                (import_specifier name: (identifier) @imported_symbol)))
            source: (string) @source_module)

        (import_statement
            (import_clause (named_imports
                (import_specifier
                    name: (identifier) @imported_symbol
                    alias: (identifier) @alias)))
            source: (string) @source_module)

        (import_statement
            (import_clause (namespace_import (identifier) @imported_symbol))
            source: (string) @source_module)
    """,
    "tsx": """
        (import_statement
            (import_clause (identifier) @imported_symbol)
            source: (string) @source_module)

        (import_statement
            (import_clause (named_imports
                (import_specifier name: (identifier) @imported_symbol)))
            source: (string) @source_module)

        (import_statement
            (import_clause (named_imports
                (import_specifier
                    name: (identifier) @imported_symbol
                    alias: (identifier) @alias)))
            source: (string) @source_module)

        (import_statement
            (import_clause (namespace_import (identifier) @imported_symbol))
            source: (string) @source_module)
    """,
}

LANGUAGE_MAP = {
    "python": Language(tspython.language()),
    "javascript": Language(tsjavascript.language()),
    "typescript": Language(tstypescript.language_typescript()),
    "tsx": Language(tstypescript.language_tsx()),
}

EXTENSION_MAP = {
    ".py": "python",
    ".js":  "javascript",
    ".jsx": "javascript",   # JSX uses the JS grammar
    ".ts":  "typescript",
    ".tsx": "tsx",   # TSX uses the TS grammar
}
