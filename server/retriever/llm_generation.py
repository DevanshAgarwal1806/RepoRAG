import os
import json
import networkx as nx
from collections import deque
from pathlib import Path
import argparse
import sys

RETRIEVER_DIR = Path(__file__).resolve().parent
SERVER_DIR = RETRIEVER_DIR.parent

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from retriever.query_expansion import expand_query
from retriever.hybrid_retrieval_dependency import hybrid_retrieval_with_dependency, load_data
from retriever.hybrid_retrieval import hybrid_retrieval

import tiktoken
_enc = tiktoken.get_encoding("cl100k_base")

def estimate_tokens(text: str) -> int:
    return len(_enc.encode(text))

def assemble_llm_context(functions_retrieved: list[tuple[str, str]], function_map: dict, output_dir: str, with_dependency: bool = False) -> str:
    # Format the payload
    llm_prompt_context = "### CODEBASE CONTEXT\n\n"
    for role, node_id in functions_retrieved:
        # Filter out external/library nodes (they don't have source code in our map)
        if node_id.startswith("__external__") or node_id not in function_map:
            continue
            
        fn_data = function_map[node_id]
        
        # Safely extract the file path and source code
        file_path = fn_data.get("file_path", fn_data.get("file", "Unknown File"))
        source_code = fn_data.get("source_code", fn_data.get("source", "No source available."))
        
        if with_dependency:
            llm_prompt_context += f"{role}: `{fn_data.get('name', 'Unknown')}`\n\n"
        else:
            llm_prompt_context += f"`{fn_data.get('name', 'Unknown')}`\n\n"
        llm_prompt_context += f"File: `{file_path}`\n\n"
        llm_prompt_context += f"Code:\n```\n{source_code}\n```\n\n"
        
    return llm_prompt_context

def llm_generation(output_dir: Path, query: str, save_prompt: bool = False, with_dependency: bool = False) -> tuple[list[tuple[str, str]], str]:
    # 1. Load the fully embedded data
    corpus_path_embeddings = output_dir / "embeddings.json"
    corpus_path_functions = output_dir / "extracted_functions.json"
    
    with open(corpus_path_embeddings, "r", encoding="utf-8") as f:
        corpus_embeddings = json.load(f)
    with open(corpus_path_functions, "r", encoding="utf-8") as f:
        corpus_functions = json.load(f)
        
    # 3. The User Input
    original_query = query.strip()

    # 4. Expand Query (LLM)
    expanded_query = expand_query(original_query)
    
    G, function_map = load_data(str(output_dir))
    
    if with_dependency:
        functions_retrieved = hybrid_retrieval_with_dependency(
            expanded_query,
            corpus_functions,
            corpus_embeddings,
            G,
            top_k=7,
        )
    else:
        functions_retrieved = hybrid_retrieval(
            expanded_query,
            corpus_functions,
            corpus_embeddings,
            top_k=7,
        )

    llm_prompt_context = assemble_llm_context(functions_retrieved, function_map, str(output_dir), d=1)
    final_payload = f"### USER QUERY: {original_query}\n\n{llm_prompt_context}"
    
    if save_prompt:
        prompt_file = output_dir / "final_llm_payload.md"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(final_payload)
            print(f"Saved to: {prompt_file}")
        
    print(f"Successfully generated full RAG payload: {len(final_payload)} characters.")
    print(f"Tokens in payload: {estimate_tokens(final_payload)}")
    return functions_retrieved, final_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Retrieval with RRF and Graph Context")
    parser.add_argument("--output", "-o", required=True, help="Output directory for results")
    parser.add_argument("--query", "-q", required=True, help="User query to search for")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    query = args.query

    llm_generation(output_dir, query, save_prompt=True)
