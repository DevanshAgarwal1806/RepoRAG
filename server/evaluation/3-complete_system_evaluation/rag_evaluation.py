from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

COMPLETE_EVALUATION_DIR = Path(__file__).resolve().parent
EVALUATION_ROOT = COMPLETE_EVALUATION_DIR.parent
SERVER_DIR = EVALUATION_ROOT.parent
GROUND_TRUTH_DIR = EVALUATION_ROOT / "0-ground_truth_construction"
DEFAULT_OUTPUT_DIR = SERVER_DIR / "sample_repository_output"
RESULTS_PATH_SINGLE = COMPLETE_EVALUATION_DIR / "rag_evaluation_results_single.json"
RESULTS_PATH_MULTI = COMPLETE_EVALUATION_DIR / "rag_evaluation_results_multi.json"

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
    
from retriever.generator import generate_rag_answer
from retriever.llm_generation import llm_generation
from retriever.query_classifier import classify_query, load_examples, build_example_store

ENV_PATH = SERVER_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def run_rag_system(query: str, output_dir: Path, classify_query_fn=None) -> dict:

    if classify_query_fn:
        query_type = classify_query_fn(query)["label"]
    else:
        query_type = "multi-hop"
    if query_type == "single-hop":
        _functions_retrieved, llm_payload = llm_generation(output_dir, query, with_dependency=False)
    else:
        _functions_retrieved, llm_payload = llm_generation(output_dir, query, with_dependency=True)
    answer = generate_rag_answer(str(output_dir), llm_payload[:41000])

    return {
        "query": query,
        "retrieved_context": llm_payload,
        "generated_answer": answer or "",
    }

def batch_rag_system(
    results_path: str,
    output_dir: str,
    query_key: str = "query",
    classify_query_fn = None
):

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
        
    num_results = 0

    out = Path(output_dir)
    for idx, item in enumerate(results):
        if item.get("generated_answer") is not None or item.get("retrieved_context") is not None:
            num_results += 1
            print(f"[{idx+1}/{len(results)}] Skipping query (already has answer/context)")
            continue
        query = item.get(query_key, "")
        print(f"[{idx+1}/{len(results)}] Running RAG system for query")
        try:
            r = run_rag_system(query, out, classify_query_fn=classify_query_fn)
            if r["generated_answer"]:
                item["retrieved_context"] = r["retrieved_context"]
                item["generated_answer"] = r["generated_answer"]
                num_results += 1
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4)
            else:
                print(f"  WARNING: No answer generated for this query.")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"[RepoRAG System] Done. {num_results}/{len(results)} results saved to {results_path}")
    
if __name__ == "__main__":
    examples = load_examples()
    store = build_example_store(examples)

    batch_rag_system(
        results_path=RESULTS_PATH_SINGLE,
        output_dir=DEFAULT_OUTPUT_DIR,
        query_key="query",
        classify_query_fn=lambda q: classify_query(q, store)
    )
    
    batch_rag_system(
        results_path=RESULTS_PATH_MULTI,
        output_dir=DEFAULT_OUTPUT_DIR,
        query_key="query",
        classify_query_fn=lambda q: classify_query(q, store)
    )
