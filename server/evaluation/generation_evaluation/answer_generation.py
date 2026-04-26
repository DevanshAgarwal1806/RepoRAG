import sys

import tiktoken
import json
from pathlib import Path
import time

GENERATOR_EVALUATION_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = GENERATOR_EVALUATION_DIR.parent
SERVER_DIR = EVALUATION_DIR.parent
GROUND_TRUTH_DIR = EVALUATION_DIR / "ground_truth_construction"
DEFAULT_OUTPUT_DIR = SERVER_DIR / "sample_repository_output"
COMPLETE_SYSTEM_EVALUATION_DIR = EVALUATION_DIR / "complete_system_evaluation"
AGENTIC_SYSTEM_OUTPUT_DIR = EVALUATION_DIR / "agentic_ai_baseline"

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
    
from retriever.generator import generate_rag_answer

def _number_of_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def generate_answers_from_models(
    results_filepath: str,
    model_name: str = "llama-3.3-70b-versatile",
    provider: str = "groq"
) -> list:
    """
    Main runner. Loads RAG and Agentic results, runs both judges on each item,
    and saves incrementally to allow resuming if interrupted.
    """
    results_path = GENERATOR_EVALUATION_DIR / results_filepath

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        print(f"Results file not found at {results_path}.")
        return []
    
    print(f"Starting answer generation for {len(results)} items...\n")
    for idx, item in enumerate(results):
        query = item.get("query", "")
        print(f"[{idx+1}/{len(results)}] Generating answer for query: {query[:70]}...")
        payload = item.get("retrieved_context", "")
        print(f"Payload token count: {_number_of_tokens(payload)}")
        if item.get("payload_token_count") is None:
            item["payload_token_count"] = _number_of_tokens(payload)

        if item.get(f"generated_answer_{model_name}") in (None, "ERROR"):
            start = time.perf_counter()
            generated_answer = generate_rag_answer(DEFAULT_OUTPUT_DIR, payload, provider, model_name)
            end = time.perf_counter()
            if len(generated_answer) > 0:
                item[f"generated_answer_{model_name}"] = generated_answer
                item[f"{model_name}_generation_time_sec"] = f"{end - start:.6f}"
                print(f"Generated {model_name} answer: {generated_answer[:70]}...")
            else:
                item[f"generated_answer_{model_name}"] = "ERROR"

        # Save after each item to allow resuming
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"Finished generating answers for all {len(results)} items.")
    return results

if __name__ == "__main__":
    RESULTS_FILE = "rag_system_results_single.json"

    generate_answers_from_models(RESULTS_FILE, "llama-3.3-70b-versatile", "groq")
    # generate_answers_from_models(RESULTS_FILE, "qwen2.5-coder:7b", "ollama")
    # generate_answers_from_models(RESULTS_FILE, "llama3.1:8b", "ollama")
    
    RESULTS_FILE = "rag_system_results_multi.json"

    generate_answers_from_models(RESULTS_FILE, "llama-3.3-70b-versatile", "groq")
    # generate_answers_from_models(RESULTS_FILE, "qwen2.5-coder:7b", "ollama")
    # generate_answers_from_models(RESULTS_FILE, "llama3.1:8b", "ollama")