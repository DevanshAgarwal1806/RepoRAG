import sys

import tiktoken
import json
from pathlib import Path
import time

BASELINE_DIR = Path(__file__).resolve().parent
SERVER_DIR = BASELINE_DIR.parent.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
    
COMPLETE_EVALUATION_DIR = Path(__file__).resolve().parent
EVALUATION_ROOT = COMPLETE_EVALUATION_DIR.parent
SERVER_DIR = EVALUATION_ROOT.parent
GROUND_TRUTH_DIR = EVALUATION_ROOT / "0-ground_truth_construction"
DEFAULT_OUTPUT_DIR = SERVER_DIR / "sample_repository_output"
DEFAULT_RESULTS_PATH_SINGLE = COMPLETE_EVALUATION_DIR / "rag_evaluation_results_single.json"
DEFAULT_RESULTS_PATH_MULTI = COMPLETE_EVALUATION_DIR / "rag_evaluation_results_multi.json"
    
from retriever.generator import generate_rag_answer
from retriever.hybrid_retrieval_dependency import load_data

if __name__ == "__main__":
    with open(SERVER_DIR / "evaluation" / "2-generation_evaluation" / "gen_results_multi.json", "r", encoding="utf-8") as f:
        results = json.load(f)
        
    rag = []
    for item in results:
        query = item.get("query", "")
        context = item.get("actual_context", "")
        
        rag.append({
            "query": query,
            "actual_context": context,
        })
    
    with open(DEFAULT_RESULTS_PATH_MULTI, "w", encoding="utf-8") as f:
        json.dump(rag, f, indent=4)