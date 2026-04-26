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
GROUND_TRUTH_DIR = EVALUATION_ROOT / "ground_truth_construction"
DEFAULT_OUTPUT_DIR = SERVER_DIR / "sample_repository_output"
DEFAULT_RESULTS_PATH = COMPLETE_EVALUATION_DIR / "rag_evaluation_results.json"

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

ENV_PATH = SERVER_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def run_rag_system(query: str, output_dir: Path) -> dict:
    from retriever.generator import generate_rag_answer
    from retriever.llm_generation import llm_generation

    _functions_retrieved, llm_payload = llm_generation(output_dir, query)
    answer = generate_rag_answer(str(output_dir), llm_payload)

    return {
        "query": query,
        "retrieved_context": llm_payload,
        "generated_answer": answer or "",
    }

def batch_rag_system(
    ground_truth_path: str,
    results_path: str,
    output_dir: str,
    ground_truth_dir: str,
    results_dir: str,
    query_key: str = "query",
):
    ground_truth_path = Path(ground_truth_dir) / ground_truth_path
    results_path = Path(results_dir) / results_path
    with open(ground_truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    try:
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
        done = {r["query"] for r in results}
        print(f"[Your System] Resuming: {len(results)} done.")
    except (FileNotFoundError, json.JSONDecodeError):
        results, done = [], set()

    out = Path(output_dir)
    for idx, item in enumerate(ground_truth):
        q = item.get(query_key, "")
        if not q or q in done:
            continue
        print(f"\n[Your System {idx+1}/{len(ground_truth)}] {q[:70]}…")
        try:
            r = run_rag_system(q, out)
            results.append(r)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"[Your System] Done. {len(results)} results saved to {results_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch RAG System Evaluation")
    parser.add_argument("--ground_truth_dir", type=str, default=str(GROUND_TRUTH_DIR), help="Path to the ground truth JSON file.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory containing the embedded corpus and where intermediate files will be saved.")
    parser.add_argument("--results_dir", type=str, default=str(COMPLETE_EVALUATION_DIR), help="Path to save the evaluation results JSON.")
    parser.add_argument("--ground_truth_path", type=str, required=True, help="Path to the ground truth JSON file.")
    parser.add_argument("--results_path", type=str, required=True, help="Path to save the evaluation results JSON.")
    args = parser.parse_args()

    batch_rag_system(
        ground_truth_path=args.ground_truth_path,
        results_path=args.results_path,
        ground_truth_dir=args.ground_truth_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        query_key="query"
    )
