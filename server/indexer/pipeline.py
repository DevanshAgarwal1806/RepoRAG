# pipeline.py
import os
import sys
import json
import argparse
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from pathlib import Path

from ast_parser import index_repository, save_functions_to_json, FunctionNode
from graph_builder import build_dependency_graph

# ── Colours ───────────────────────────────────────────────────────────────────
NODE_COLORS = {
    "python":     "#4B8BBE",
    "javascript": "#F7DF1E",
    "typescript": "#3178C6",
    "java":       "#ED8B00",
    "go":         "#00ADD8",
    "rust":       "#DEA584",
    "cpp":        "#659AD2",
    "c":          "#A8B9CC",
    "ruby":       "#CC342D",
    "php":        "#8892BF",
    "external":   "#AAAAAA",
}
EDGE_COLORS = {
    "decorator": "#FF6B6B",
    "high":      "#2ECC71",   # weight >= 0.8
    "medium":    "#F39C12",   # weight >= 0.4
    "low":       "#95A5A6",   # weight <  0.4
}

def edge_color(data: dict) -> str:
    if data.get("is_decorator"):
        return EDGE_COLORS["decorator"]
    w = data.get("weight", 0)
    if w >= 0.8:
        return EDGE_COLORS["high"]
    if w >= 0.4:
        return EDGE_COLORS["medium"]
    return EDGE_COLORS["low"]

# ── Graph stats ────────────────────────────────────────────────────────────────
def print_graph_stats(G: nx.DiGraph):
    internal = [n for n, d in G.nodes(data=True) if d.get("language") != "external"]
    external = [n for n, d in G.nodes(data=True) if d.get("language") == "external"]

    print("\n" + "="*60)
    print("DEPENDENCY GRAPH STATS")
    print("="*60)
    print(f"  Total nodes      : {G.number_of_nodes()}")
    print(f"    Internal        : {len(internal)}")
    print(f"    External (libs) : {len(external)}")
    print(f"  Total edges      : {G.number_of_edges()}")

    if internal:
        # Top 5 most-called internal functions
        in_degrees = [(n, G.in_degree(n)) for n in internal]
        top_called = sorted(in_degrees, key=lambda x: x[1], reverse=True)[:5]
        print("\n  Top 5 most-called internal functions:")
        for node_id, deg in top_called:
            name = G.nodes[node_id].get("name", node_id)
            print(f"    [{deg:>3} callers]  {name}")

        # Top 5 functions that call the most others
        out_degrees = [(n, G.out_degree(n)) for n in internal]
        top_callers = sorted(out_degrees, key=lambda x: x[1], reverse=True)[:5]
        print("\n  Top 5 functions that call the most others:")
        for node_id, deg in top_callers:
            name = G.nodes[node_id].get("name", node_id)
            print(f"    [{deg:>3} callees]  {name}")

    if external:
        # Most-used external calls
        ext_in = [(n, G.in_degree(n)) for n in external]
        top_ext = sorted(ext_in, key=lambda x: x[1], reverse=True)[:5]
        print("\n  Top 5 most-used external calls:")
        for node_id, deg in top_ext:
            name = G.nodes[node_id].get("name", node_id)
            print(f"    [{deg:>3} callers]  {name}")

# ── Visualization ──────────────────────────────────────────────────────────────
def visualize_graph(G: nx.DiGraph, output_path: str = "dependency_graph.png",
                    max_nodes: int = 80):
    if G.number_of_nodes() == 0:
        print("Graph is empty — nothing to visualize.")
        return

    if G.number_of_nodes() > max_nodes:
        print(f"  Graph has {G.number_of_nodes()} nodes — trimming to top {max_nodes} by degree.")
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    n = G.number_of_nodes()

    # Scale figure size DOWN and node size UP based on node count
    fig_size  = max(10, min(20, 10 + n * 0.1))   # 10–20 inches, not 24
    node_base = max(800, 2400 - n * 18)           # shrinks slightly as graph grows
    font_size = max(8, 13 - n // 15)              # 8–13pt

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.75))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    if n <= 30:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=2.5, iterations=60, seed=42)

    node_color_list = [
        NODE_COLORS.get(G.nodes[n].get("language", "external"), "#AAAAAA")
        for n in G.nodes()
    ]
    node_size_list = [
        node_base + 120 * G.degree(n) for n in G.nodes()
    ]

    edge_color_list = [edge_color(d) for _, _, d in G.edges(data=True)]
    edge_width_list = [
        1.5 + d.get("weight", 0.1) * 1.5 for _, _, d in G.edges(data=True)
    ]

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_color_list,
        width=edge_width_list,
        arrows=True,
        arrowsize=15,
        arrowstyle="-|>",
        alpha=0.7,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_color_list,
        node_size=node_size_list,
        alpha=0.95,
        linewidths=1.5,
        edgecolors="#FFFFFF",
    )
    labels = {
        n: G.nodes[n].get("name", n).split(".")[-1]
        for n in G.nodes()
    }
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=font_size,
        font_color="#FFFFFF",
        font_weight="bold",
    )

    lang_patches = [
        mpatches.Patch(color=color, label=lang.capitalize())
        for lang, color in NODE_COLORS.items()
        if any(G.nodes[n].get("language") == lang for n in G.nodes())
    ]
    edge_patches = [
        mpatches.Patch(color=EDGE_COLORS["decorator"], label="Decorator call"),
        mpatches.Patch(color=EDGE_COLORS["high"],      label="High confidence (≥0.8)"),
        mpatches.Patch(color=EDGE_COLORS["medium"],    label="Medium confidence (≥0.4)"),
        mpatches.Patch(color=EDGE_COLORS["low"],       label="Low confidence (<0.4)"),
    ]
    ax.legend(
        handles=lang_patches + edge_patches,
        loc="upper left",
        fontsize=9,
        facecolor="#1E1E2E",
        edgecolor="#444466",
        labelcolor="white",
        framealpha=0.85,
    )

    ax.set_title("Function Dependency Graph", color="white", fontsize=14, pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved graph visualization → {output_path}")

# ── Pipeline ───────────────────────────────────────────────────────────────────
def run_pipeline(repo_path: str, output_dir: str = "."):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Parse ──────────────────────────────────────────────────────────
    print(f"\n[1/3] Parsing repository: {repo_path}")
    functions, global_import_map = index_repository(repo_path)

    if not functions:
        print("No functions extracted. Exiting.")
        return

    functions_json = output_dir / "extracted_functions.json"
    save_functions_to_json(functions, str(functions_json))
    print(f"  Saved {len(functions)} functions → {functions_json}")

    imports_json = output_dir / "import_map.json"
    with open(imports_json, "w", encoding="utf-8") as f:
        json.dump(global_import_map, f, indent=2, ensure_ascii=False)
    print(f"  Saved import map → {imports_json}")

    # ── Step 2: Build graph ────────────────────────────────────────────────────
    print(f"\n[2/3] Building dependency graph ...")
    G = build_dependency_graph(functions, global_import_map)
    print_graph_stats(G)

    graph_json = output_dir / "dependency_graph.json"
    nx.node_link_data(G)
    with open(graph_json, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved graph JSON → {graph_json}")

    # ── Step 3: Visualize ──────────────────────────────────────────────────────
    print(f"\n[3/3] Visualizing dependency graph ...")
    graph_png = output_dir / "dependency_graph.png"
    visualize_graph(G, output_path=str(graph_png))

    print("\n" + "="*60)
    print("Pipeline complete.")
    print(f"  Functions JSON : {functions_json}")
    print(f"  Import map     : {imports_json}")
    print(f"  Graph JSON     : {graph_json}")
    print(f"  Graph PNG      : {graph_png}")
    print("="*60)

    return G

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AST parser → dependency graph pipeline")
    parser.add_argument("repo_path", nargs="?", default=".", 
                        help="Path to the repository to analyse (default: current dir)")
    parser.add_argument("--output", "-o", default="output",
                        help="Directory to write all output files (default: ./output)")
    args = parser.parse_args()

    run_pipeline(args.repo_path, args.output)