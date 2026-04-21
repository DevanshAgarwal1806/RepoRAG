import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

from retriever.generator import generate_rag_answer
from indexer.pipeline import run_indexer_pipeline
from retriever.hybrid_retrieval import run_hybrid_retrieval

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
    
    if args.rerun or not all([embeddings.exists(), dependency_graph.exists(), functions_json.exists(), import_map_json.exists()]):
        run_indexer_pipeline(str(repo_dir), str(output_dir))
        print("")
    
    run_hybrid_retrieval(output_dir, args.query)
    print("")
    generate_rag_answer(output_dir)