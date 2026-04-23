import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
import tree_sitter_java as tsjava
import tree_sitter_cpp as tscpp
import tree_sitter_c as tsc
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
                (_)*) @body
            )
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
                        body: (statement_block) @body
                    )
        )
        
        (variable_declarator
            name: (identifier) @name
            value: (arrow_function
                        body: (statement_block) @body
                    )
        )
        
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
    "java": """
        (method_declaration
            name: (identifier) @name
            body: (block) @body)

        (constructor_declaration
            name: (identifier) @name
            body: (constructor_body) @body)
    """,
    "c": """
        (function_definition
            declarator: (function_declarator
                            declarator: (identifier) @name
                        )
            body: (compound_statement) @body
        )

        (function_definition
            declarator: (pointer_declarator
                            declarator: (function_declarator
                                            declarator: (identifier) @name
                                        )
                        )
            body: (compound_statement) @body
        )
    """,
    "cpp": """
        (function_definition
            declarator: (function_declarator
                declarator: (identifier) @name)
            body: (compound_statement) @body)

        (function_definition
            declarator: (function_declarator
                declarator: (field_identifier) @name)
            body: (compound_statement) @body)

        (function_definition
            declarator: (function_declarator
                declarator: (qualified_identifier) @name)
            body: (compound_statement) @body)

        (function_definition
            declarator: (pointer_declarator
                declarator: (function_declarator
                    declarator: (identifier) @name))
            body: (compound_statement) @body)
    """,
}

CALL_QUERIES = {
    "python": """
        (call 
            function: 
                (identifier) @callee
        )
        (call 
            function: 
                (attribute 
                    object: (_) @receiver
                    attribute: (identifier) @callee
                )
        )
    """,
    "javascript": """
        (call_expression 
            function: (identifier) @callee
        )
        (call_expression 
            function: 
                (member_expression 
                    object: (_) @receiver
                    property: (property_identifier) @callee
                )
        )
    """,
    "typescript": """
        (call_expression 
            function: (identifier) @callee
        )
        (call_expression 
            function: 
                (member_expression 
                    object: (_) @receiver
                    property: (property_identifier) @callee
                )
        )
    """,
    "tsx": """
        (call_expression 
            function: (identifier) @callee
        )
        (call_expression 
            function: 
                (member_expression 
                    object: (_) @receiver
                    property: (property_identifier) @callee
                )
        )
    """,
    "java": """
        (method_invocation
            name: (identifier) @callee)
        
        (method_invocation
            object: (_) @receiver
            name: (identifier) @callee)

        (object_creation_expression
            type: (type_identifier) @callee)
    """,
    "c": """
        (call_expression
            function: (identifier) @callee)

        (call_expression
            function: (field_expression
                argument: (_) @receiver
                field: (field_identifier) @callee))
                
        (call_expression
            function: (parenthesized_expression
                            (pointer_expression
                                (field_expression
                                    field: (field_identifier) @callee
                                )
                            )
                        )
        )
        
        (call_expression
            function: (parenthesized_expression
                            (pointer_expression
                                (identifier) @callee
                            )
                        )
        )
    """,
    "cpp": """
        (call_expression
            function: (identifier) @callee)

        (call_expression
            function: (field_expression
                argument: (_) @receiver
                field: (field_identifier) @callee))

        (call_expression
            function: (qualified_identifier
                scope: (_) @receiver
                name: (identifier) @callee))
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
    "java": """
        (import_declaration
            (scoped_identifier
                scope: (_) @source_module
                name: (identifier) @imported_symbol))

        (import_declaration
            (scoped_identifier
                scope: (_) @source_module
                name: (asterisk) @wildcard))
    """,
    "c": """
        (preproc_include
            path: (system_lib_string) @source_module)

        (preproc_include
            path: (string_literal) @source_module)
    """,
    "cpp": """
        (preproc_include
            path: (system_lib_string) @source_module)

        (preproc_include
            path: (string_literal) @source_module)
    """,
}

LANGUAGE_MAP = {
    "python": Language(tspython.language()),
    "javascript": Language(tsjavascript.language()),
    "typescript": Language(tstypescript.language_typescript()),
    "tsx": Language(tstypescript.language_tsx()),
    "java": Language(tsjava.language()),
    "c": Language(tsc.language()),
    "cpp": Language(tscpp.language()),
}

EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",   # JSX uses the JS grammar
    ".ts": "typescript",
    ".tsx": "tsx",   # TSX uses the TS grammar
    ".java": "java",
    ".c": "c",
    ".h": "c",       # C headers — may also contain function definitions
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
}
