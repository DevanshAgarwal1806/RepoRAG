from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
from dotenv import load_dotenv

EVALUATION_DIR = Path(__file__).resolve().parent
SERVER_DIR = EVALUATION_DIR.parent
GROUND_TRUTH_DIR = EVALUATION_DIR / "0-ground_truth_construction"
DEFAULT_OUTPUT_DIR = SERVER_DIR / "sample_repository_output"
DEFAULT_RESULTS_PATH = EVALUATION_DIR / "retrieval_evaluation_results.json"

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

ENV_PATH = SERVER_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from retriever.bm25_basic import bm25_search
from retriever.bm25_dependency import bm25_dependency
from retriever.hybrid_retrieval import hybrid_retrieval
from retriever.hybrid_retrieval_dependency import (
    hybrid_retrieval_with_dependency,
)

STRATEGIES = ("bm25", "bm25_dep", "bm25_dense", "bm25_dense_dep")
K_VALUES = (3, 5, 7)


@dataclass(frozen=True)
class SingleHopQuery:
    query_type: str
    query: str
    target_file: str
    node_id: str


@dataclass(frozen=True)
class MultiHopQuery:
    query: str
    root_id: str
    required_functions: list[str]


@dataclass(frozen=True)
class PerQueryResult:
    query: str
    strategy: str
    relevant: list[str]
    retrieved: list[str]
    reciprocal_rank: float | None
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required evaluation input not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dependency_graph(path: Path) -> nx.DiGraph:
    graph_data = load_json(path)
    edge_key = "links" if "links" in graph_data else "edges"
    try:
        return nx.node_link_graph(graph_data, edges=edge_key)
    except TypeError:
        return nx.node_link_graph(graph_data, link=edge_key)


def load_single_hop_queries(path: Path) -> list[SingleHopQuery]:
    queries: list[SingleHopQuery] = []
    for idx, item in enumerate(load_json(path), start=1):
        query_type = item.get("query_type", item.get("queryType"))
        missing = [
            key
            for key, value in {
                "query_type/queryType": query_type,
                "query": item.get("query"),
                "target_file": item.get("target_file"),
                "node_id": item.get("node_id"),
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Single-hop query #{idx} is missing: {', '.join(missing)}")

        queries.append(
            SingleHopQuery(
                query_type=query_type,
                query=item["query"],
                target_file=item["target_file"],
                node_id=item["node_id"],
            )
        )
    return queries


def load_multi_hop_queries(path: Path) -> list[MultiHopQuery]:
    queries: list[MultiHopQuery] = []
    fallback_count = 0
    for idx, item in enumerate(load_json(path), start=1):
        query = item.get("query")
        root_id = item.get("root_id")
        if not query or not root_id:
            raise ValueError(f"Multi-hop query #{idx} is missing query or root_id")

        required_functions = item.get("required_functions")
        if not required_functions:
            required_functions = [root_id]
            fallback_count += 1

        queries.append(
            MultiHopQuery(
                query=query,
                root_id=root_id,
                required_functions=required_functions,
            )
        )

    if fallback_count:
        print(
            f"[Warning] {fallback_count} multi-hop queries did not declare "
            "required_functions; using root_id as the only relevant function."
        )
    return queries


def validate_embeddings(embeddings: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(embeddings, list):
        raise ValueError(f"{path} must contain a list of embedding records.")

    invalid = [
        item.get("id", f"record #{idx}")
        for idx, item in enumerate(embeddings, start=1)
        if not isinstance(item, dict) or not isinstance(item.get("embedding"), list)
    ]
    if invalid:
        examples = ", ".join(str(item) for item in invalid[:3])
        raise ValueError(
            f"{path} contains malformed embeddings. Expected each embedding to be "
            f"a vector list, but found scalar/missing values for: {examples}. "
            "Regenerate embeddings with the fixed indexer before running dense evaluation."
        )
    return embeddings


def warn_about_missing_ground_truth(
    function_ids: set[str],
    single_hop_queries: list[SingleHopQuery],
    multi_hop_queries: list[MultiHopQuery],
) -> None:
    missing_single = [query.node_id for query in single_hop_queries if query.node_id not in function_ids]
    missing_multi = sorted(
        {
            fn_id
            for query in multi_hop_queries
            for fn_id in query.required_functions
            if fn_id not in function_ids
        }
    )

    if missing_single:
        print(
            f"[Warning] {len(missing_single)} single-hop ground-truth node IDs "
            "are not present in extracted_functions.json."
        )
    if missing_multi:
        print(
            f"[Warning] {len(missing_multi)} multi-hop ground-truth function IDs "
            "are not present in extracted_functions.json."
        )


def reciprocal_rank(retrieved_ids: list[str], relevant_set: set[str]) -> float:
    for index, function_id in enumerate(retrieved_ids, start=1):
        if function_id in relevant_set:
            return 1.0 / index
    return 0.0


def recall_at_k(retrieved_ids: list[str], relevant_set: set[str], k: int) -> float:
    if not relevant_set:
        return 0.0
    hits = sum(1 for function_id in retrieved_ids[:k] if function_id in relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(retrieved_ids: list[str], relevant_set: set[str], k: int) -> float:
    dcg = 0.0
    for index, function_id in enumerate(retrieved_ids[:k], start=1):
        if function_id in relevant_set:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


class RetrievalPipeline:
    def __init__(
        self,
        extracted_functions: list[dict[str, Any]],
        embeddings: list[dict[str, Any]],
        dependency_graph: nx.DiGraph,
    ) -> None:
        self.extracted_functions = extracted_functions
        self.embeddings = embeddings
        self.dependency_graph = dependency_graph

    def bm25(self, query: str, k: int) -> list[str]:
        return [doc["id"] for doc, _ in bm25_search(query, self.extracted_functions, k)]

    def bm25_dep(self, query: str, k: int) -> list[str]:
        return [
            function_id
            for _, function_id in bm25_dependency(
                query,
                self.extracted_functions,
                self.dependency_graph,
                k,
            )
        ]

    def bm25_dense(self, query: str, k: int) -> list[str]:
        return [
            function_id
            for function_id, _ in hybrid_retrieval(
                query,
                self.extracted_functions,
                self.embeddings,
                top_k=k,
            )
        ]

    def bm25_dense_dep(self, query: str, k: int) -> list[str]:
        return [
            function_id
            for _, function_id in hybrid_retrieval_with_dependency(
                query,
                self.extracted_functions,
                self.embeddings,
                self.dependency_graph,
                k,
            )
        ]


class Evaluator:
    def __init__(self, pipeline: RetrievalPipeline, top_k: int = 10) -> None:
        self.pipeline = pipeline
        self.top_k = max(top_k, max(K_VALUES))
        self.retrieval: dict[str, Callable[[str, int], list[str]]] = {
            "bm25": pipeline.bm25,
            "bm25_dep": pipeline.bm25_dep,
            "bm25_dense": pipeline.bm25_dense,
            "bm25_dense_dep": pipeline.bm25_dense_dep,
        }

    def evaluate_single_hop(
        self,
        queries: list[SingleHopQuery],
    ) -> dict[str, list[PerQueryResult]]:
        results: dict[str, list[PerQueryResult]] = {strategy: [] for strategy in STRATEGIES}
        total = len(queries)
        for index, query in enumerate(queries, start=1):
            print(f"  [Single-hop {index}/{total}] {query.query[:80]}")
            relevant = {query.node_id}
            for strategy, retrieval_fn in self.retrieval.items():
                retrieved_ids = retrieval_fn(query.query, self.top_k)
                results[strategy].append(
                    PerQueryResult(
                        query=query.query,
                        strategy=strategy,
                        relevant=[query.node_id],
                        retrieved=retrieved_ids,
                        reciprocal_rank=reciprocal_rank(retrieved_ids, relevant),
                        recall_at_k={
                            k: recall_at_k(retrieved_ids, relevant, k) for k in K_VALUES
                        },
                        ndcg_at_k={
                            k: ndcg_at_k(retrieved_ids, relevant, k) for k in K_VALUES
                        },
                    )
                )
        return results

    def evaluate_multi_hop(
        self,
        queries: list[MultiHopQuery],
    ) -> dict[str, list[PerQueryResult]]:
        results: dict[str, list[PerQueryResult]] = {strategy: [] for strategy in STRATEGIES}
        total = len(queries)
        for index, query in enumerate(queries, start=1):
            print(f"  [Multi-hop {index}/{total}] {query.query[:80]}")
            relevant = set(query.required_functions)
            for strategy, retrieval_fn in self.retrieval.items():
                retrieved_ids = retrieval_fn(query.query, self.top_k)
                results[strategy].append(
                    PerQueryResult(
                        query=query.query,
                        strategy=strategy,
                        relevant=query.required_functions,
                        retrieved=retrieved_ids,
                        reciprocal_rank=None,
                        recall_at_k={
                            k: recall_at_k(retrieved_ids, relevant, k) for k in K_VALUES
                        },
                        ndcg_at_k={
                            k: ndcg_at_k(retrieved_ids, relevant, k) for k in K_VALUES
                        },
                    )
                )
        return results

    @staticmethod
    def aggregate(
        results: dict[str, list[PerQueryResult]],
        include_mrr: bool = True,
    ) -> dict[str, dict[str, float]]:
        aggregate_results: dict[str, dict[str, float]] = {}
        for strategy, per_query in results.items():
            if not per_query:
                continue

            strategy_metrics: dict[str, float] = {}
            if include_mrr:
                rr_values = [
                    result.reciprocal_rank
                    for result in per_query
                    if result.reciprocal_rank is not None
                ]
                strategy_metrics["MRR"] = float(np.mean(rr_values)) if rr_values else 0.0

            for k in K_VALUES:
                strategy_metrics[f"Recall@{k}"] = float(
                    np.mean([result.recall_at_k[k] for result in per_query])
                )
                strategy_metrics[f"nDCG@{k}"] = float(
                    np.mean([result.ndcg_at_k[k] for result in per_query])
                )

            aggregate_results[strategy] = strategy_metrics
        return aggregate_results


def serialize_per_query_results(
    results: dict[str, list[PerQueryResult]],
    single_hop_queries: list[SingleHopQuery] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    query_type_by_text = {
        query.query: query.query_type for query in single_hop_queries or []
    }

    serialized: dict[str, list[dict[str, Any]]] = {}
    for strategy, per_query in results.items():
        serialized[strategy] = []
        for result in per_query:
            item: dict[str, Any] = {
                "query": result.query,
                "relevant": result.relevant,
                "retrieved_top_k": result.retrieved,
                "reciprocal_rank": result.reciprocal_rank,
                "recall_at_k": result.recall_at_k,
                "ndcg_at_k": result.ndcg_at_k,
            }
            if single_hop_queries is not None:
                item["query_type"] = query_type_by_text.get(result.query, "unknown")
            serialized[strategy].append(item)
    return serialized


def run_evaluation(output_dir: Path, results_path: Path, top_k: int) -> None:
    single_hop_path = GROUND_TRUTH_DIR / "singlehop_ground_truth.json"
    multi_hop_path = GROUND_TRUTH_DIR / "multihop_ground_truth.json"

    single_hop_queries = load_single_hop_queries(single_hop_path)
    multi_hop_queries = load_multi_hop_queries(multi_hop_path)

    extracted_functions = load_json(output_dir / "extracted_functions.json")
    embeddings_path = output_dir / "embeddings.json"
    embeddings = validate_embeddings(load_json(embeddings_path), embeddings_path)
    dependency_graph = load_dependency_graph(output_dir / "dependency_graph.json")

    function_ids = {
        item["id"]
        for item in extracted_functions
        if isinstance(item, dict) and "id" in item
    }
    warn_about_missing_ground_truth(function_ids, single_hop_queries, multi_hop_queries)

    pipeline = RetrievalPipeline(extracted_functions, embeddings, dependency_graph)
    evaluator = Evaluator(pipeline, top_k=top_k)

    print("\nEvaluating single-hop queries ...")
    single_hop_results = evaluator.evaluate_single_hop(single_hop_queries)
    single_hop_aggregate = evaluator.aggregate(single_hop_results, include_mrr=True)

    print("\nEvaluating multi-hop queries ...")
    multi_hop_results = evaluator.evaluate_multi_hop(multi_hop_queries)
    multi_hop_aggregate = evaluator.aggregate(multi_hop_results, include_mrr=False)

    output = {
        "metadata": {
            "output_dir": str(output_dir),
            "top_k": max(top_k, max(K_VALUES)),
            "strategies": list(STRATEGIES),
            "k_values": list(K_VALUES),
            "single_hop_queries": len(single_hop_queries),
            "multi_hop_queries": len(multi_hop_queries),
        },
        "single_hop": {
            "aggregate": single_hop_aggregate,
            "per_query": serialize_per_query_results(
                single_hop_results,
                single_hop_queries,
            ),
        },
        "multi_hop": {
            "aggregate": multi_hop_aggregate,
            "per_query": serialize_per_query_results(multi_hop_results),
        },
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n[Done] Full results saved to {results_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RepoRAG retrieval strategies.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing extracted_functions.json, embeddings.json, and dependency_graph.json.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path where retrieval evaluation results should be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of retrieved IDs to keep per query.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  RepoRAG Retrieval Evaluation")
    print("=" * 70)

    args = parse_args()
    run_evaluation(
        output_dir=args.output_dir,
        results_path=args.results,
        top_k=args.top_k,
    )