# pipeline.py
import json
import argparse
import networkx as nx
from pathlib import Path
import warnings

from indexer.ast_parser import index_repository, save_functions_to_json, FunctionNode
from indexer.graph_builder import build_dependency_graph
from indexer.embedder import embed_functions

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def run_indexer_pipeline(repo_path: str, output_dir: str = "."):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse Repository
    print(f"Parsing repository: {repo_path}")
    functions, global_import_map = index_repository(repo_path)

    if not functions:
        print("No functions extracted. Exiting.")
        return

    functions_json = output_dir / "extracted_functions.json"
    save_functions_to_json(functions, str(functions_json))
    print(f"Saved {len(functions)} functions to {functions_json}")

    imports_json = output_dir / "import_map.json"
    with open(imports_json, "w", encoding="utf-8") as f:
        json.dump(global_import_map, f, indent=2, ensure_ascii=False)
    print(f"Saved import map to {imports_json}")

    # Dependency Graph
    print(f"\nBuilding dependency graph")
    G = build_dependency_graph(functions, global_import_map)

    graph_json = output_dir / "dependency_graph.json"
    nx.node_link_data(G)
    with open(graph_json, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G), f, indent=2, ensure_ascii=False)
    print(f"\nSaved graph JSON to {graph_json}")
    
    # Embedding Functions
    print(f"\nGenerating function embeddings")
    embeddings_json = output_dir / "embeddings.json"
    embed_functions(functions, output_path=str(embeddings_json))
    
    print("\nPipeline complete.")
    return G

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AST parser → dependency graph pipeline")
    parser.add_argument("repo_path", nargs="?", default=".", 
                        help="Path to the repository to analyse (default: current dir)")
    parser.add_argument("--output", "-o", default="output",
                        help="Directory to write all output files (default: ./output)")
    args = parser.parse_args()

    run_indexer_pipeline(args.repo_path, args.output)