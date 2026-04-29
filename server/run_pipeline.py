import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import sys

SERVER_DIR = Path(__file__).resolve().parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from indexer_pipeline import run_indexer_pipeline
from retriever.generator import generate_rag_answer
from retriever.hybrid_retrieval import hybrid_retrieval
from retriever.llm_generation import llm_generation
from retriever.query_classifier import classify_query, load_examples, build_example_store

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete RepoRAG pipeline: indexing, retrieval, and generation")
    parser.add_argument("--repo", "-r", required=True, help="Path to the repository to analyse")
    parser.add_argument("--rerun", action="store_true", help="Rerun the indexer")
    parser.add_argument("--query", "-q", required=True, help="User query to search for")
    args = parser.parse_args()
    
    repo_dir = Path(args.repo)
    output_dir = repo_dir.parent / f"{repo_dir.name}_output"
    
    embeddings = output_dir / "embeddings.json"
    dependency_graph = output_dir / "dependency_graph.json"
    functions_json = output_dir / "extracted_functions.json"
    import_map_json = output_dir / "import_map.json"
    
    examples = load_examples()
    store = build_example_store(examples)
    
    if args.rerun or not all([embeddings.exists(), dependency_graph.exists(), functions_json.exists(), import_map_json.exists()]):
        run_indexer_pipeline(str(repo_dir), str(output_dir))
        print("")
        
    r = run_rag_system(args.query, output_dir, classify_query_fn=lambda q: classify_query(q, store))
    print(f"\n[RepoRAG System] Retrieved Context: {r['retrieved_context']}")
    print("\n[RepoRAG System] Generated Answer:")
    print(r["generated_answer"])
