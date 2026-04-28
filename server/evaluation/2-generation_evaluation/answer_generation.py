import sys

import tiktoken
import json
from pathlib import Path
import time

GENERATOR_EVALUATION_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = GENERATOR_EVALUATION_DIR.parent
SERVER_DIR = EVALUATION_DIR.parent
GROUND_TRUTH_DIR = EVALUATION_DIR / "0-ground_truth_construction"
DEFAULT_OUTPUT_DIR = SERVER_DIR / "sample_repository_output"
COMPLETE_SYSTEM_EVALUATION_DIR = EVALUATION_DIR / "complete_system_evaluation"
AGENTIC_SYSTEM_OUTPUT_DIR = EVALUATION_DIR / "agentic_ai_baseline"

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
    
from retriever.generator import generate_rag_answer
from retriever.hybrid_retrieval_dependency import load_data

def _number_of_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def generate_answers_from_models(
    ground_truth_filepath: str,
    function_map: dict,
    single_hop: bool,
    results_filepath: str,
    model_name: str = "llama-3.3-70b-versatile",
    provider: str = "groq"
) -> list:
    """
    Main runner. Loads RAG and Agentic results, runs both judges on each item,
    and saves incrementally to allow resuming if interrupted.
    """
    results_path = GENERATOR_EVALUATION_DIR / results_filepath
    ground_truth_filepath = GROUND_TRUTH_DIR / ground_truth_filepath
    
    with open(ground_truth_filepath, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []
        print("Building initial results structure from ground truth")
        for item in ground_truth:
            actual_context = "### CODEBASE CONTEXT\n\n"
            if single_hop:
                function = fn_map.get(item["node_id"])
                actual_context += f"`{function.get('name', 'Unknown')}`\n\n"
                actual_context += f"File: `{function.get('file_path')}`\n\n"
                actual_context += f"Code:\n```\n{function.get('source_code')}\n```\n\n"
            else:
                # Multi-hop:
                root_id = item["root_id"]
                root_fn = fn_map.get(root_id)
                actual_context += f"`ROLE: PRIMARY MATCH, NAME: {root_fn.get('name', 'Unknown')}`\n\n"
                actual_context += f"File: `{root_fn.get('file_path')}`\n\n"
                actual_context += f"Code:\n```\n{root_fn.get('source_code')}\n```\n\n"
                for fn_id in item.get("required_functions", []):
                    fn = fn_map.get(fn_id)
                    actual_context += f"`ROLE: NEIGHBORING CONTEXT, NAME: {fn.get('name', 'Unknown')}`\n\n"
                    actual_context += f"File: `{fn.get('file_path')}`\n\n"
                    actual_context += f"Code:\n```\n{fn.get('source_code')}\n```\n\n"

            results.append({
                "query": item.get("query", ""),
                "actual_context": actual_context
            })
            
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"Starting answer generation for {len(results)} items...\n")
    for idx, item in enumerate(results):
        query = item.get("query", "")
        print(f"[{idx+1}/{len(results)}] Generating answer for query: {query[:70]}...")
        
        payload = item.get("actual_context", "")
        print(f"Payload token count: {_number_of_tokens(payload)}")
        if item.get("payload_token_count") is None:
            item["payload_token_count"] = _number_of_tokens(payload)

        if item.get(model_name) in (None, "ERROR"):
            start = time.perf_counter()
            generated_answer = generate_rag_answer(DEFAULT_OUTPUT_DIR, payload, provider, model_name)
            end = time.perf_counter()
            if len(generated_answer) > 0:
                item[model_name] = {"answer": generated_answer, "generation_time_sec": f"{end - start:.6f}"}
                print(f"Generated {model_name} answer")
            else:
                item[model_name] = "ERROR"

        # Save after each item to allow resuming
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"Finished generating answers for all {len(results)} items.")
    return results

if __name__ == "__main__":
    G, fn_map = load_data(DEFAULT_OUTPUT_DIR)
    RESULTS_FILE = "gen_results_single.json"

    generate_answers_from_models("singlehop_ground_truth.json", fn_map, True, RESULTS_FILE, "phi4-mini:3.8b", "ollama")
    generate_answers_from_models("singlehop_ground_truth.json", fn_map, True, RESULTS_FILE, "qwen2.5-coder:3b", "ollama")
    generate_answers_from_models("singlehop_ground_truth.json", fn_map, True, RESULTS_FILE, "gemma3:4b", "ollama")
    
    # generate_answers_from_models("singlehop_ground_truth.json", fn_map, True, RESULTS_FILE, "qwen2.5-coder:7b", "ollama")
    # generate_answers_from_models("singlehop_ground_truth.json", fn_map, True, RESULTS_FILE, "mistral:7b", "ollama")
    
    RESULTS_FILE = "gen_results_multi.json"

    generate_answers_from_models("multihop_ground_truth.json", fn_map, False, RESULTS_FILE, "phi4-mini", "ollama")
    generate_answers_from_models("multihop_ground_truth.json", fn_map, False, RESULTS_FILE, "qwen2.5-coder:3b", "ollama")
    generate_answers_from_models("multihop_ground_truth.json", fn_map, False, RESULTS_FILE, "gemma3:4b", "ollama")
    
    # generate_answers_from_models("multihop_ground_truth.json", fn_map, True, RESULTS_FILE, "qwen2.5-coder:7b", "ollama")
    # generate_answers_from_models("multihop_ground_truth.json", fn_map, True, RESULTS_FILE, "mistral:7b", "ollama")