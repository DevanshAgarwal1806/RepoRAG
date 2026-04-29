import os
import sys
import json
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv

BASELINE_DIR = Path(__file__).resolve().parent
SERVER_DIR = BASELINE_DIR.parent.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
    
GROUND_TRUTH_DIR = SERVER_DIR / "evaluation" / "0-ground_truth_construction"
SINGLEHOP_GROUND_TRUTH_PATH = GROUND_TRUTH_DIR / "singlehop_ground_truth.json"
MULTIHOP_GROUND_TRUTH_PATH = GROUND_TRUTH_DIR / "multihop_ground_truth.json"
SAMPLE_REPO_OUTPUT_DIR = SERVER_DIR / "sample_repository_output"
AGENTIC_RAG_OUTPUT_PATH_SINGLEHOP = SERVER_DIR / "evaluation" / "4-agentic_rag_baseline" / "agentic_rag_results_single.json"
AGENTIC_RAG_OUTPUT_PATH_MULTIHOP = SERVER_DIR / "evaluation" / "4-agentic_rag_baseline" / "agentic_rag_results_multi.json"

for candidate in [SERVER_DIR / ".env"]:
    if candidate.exists():
        load_dotenv(dotenv_path=candidate)
        break
    
from rank_bm25 import BM25Okapi
from retriever.bm25_basic import tokenize_code
from retriever.dense_retrieval import get_dense_rankings
from retriever.query_expansion import expand_query
from retriever.hybrid_retrieval_dependency import load_data, get_neighborhood
from retriever.hybrid_retrieval import calculate_rrf

MAX_STEPS = 5 # max agent iterations before forcing generation
TOP_K = 5 # results returned per hybrid_search call
MIN_WEIGHT = 0.4 # graph edge weight threshold (matches graph_context defaults)
MODEL_NAME = "llama-3.3-70b-versatile"

groq_client = Groq(api_key=os.getenv("ANSWER_GENERATION_LLM_KEY"))

def _serialize_context(ctx: list[dict]) -> str:
    parts = []
    for item in ctx:
        header = f"[{item.get('type')}] {item.get('name')} ({item.get('file')})"
        body   = item.get("source") or item.get("snippet") or ""
        parts.append(f"{header}\n{body}")
    return ("\n" + "=" * 60 + "\n").join(parts)

class AgentTools:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.G, self.fn_map = load_data(str(output_dir))
        with open(output_dir / "embeddings.json", encoding="utf-8") as f:
            self.corpus_embeddings = json.load(f)
            
        self.fn_ids = list(self.fn_map.keys())
        tokenized = [tokenize_code(self.fn_map[fid].get("source_code", "")) for fid in self.fn_ids]
        self.bm25 = BM25Okapi(tokenized)

    def hybrid_search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        expanded = expand_query(query)
        bm25_scores = self.bm25.get_scores(tokenize_code(expanded))
        bm25_ranked = sorted(
            zip(self.fn_ids, bm25_scores), key=lambda x: x[1], reverse=True
        )
        bm25_ranks = {fid: r for r, (fid, _) in enumerate(bm25_ranked, 1)}
        
        dense_results = get_dense_rankings(expanded, self.corpus_embeddings)
        dense_ranks = {fid: rank for fid, _, rank in dense_results}
        fused = calculate_rrf(bm25_ranks, dense_ranks)

        results = []
        for fid, rrf_score in fused[:top_k]:
            fn = self.fn_map.get(fid, {})
            src = fn.get("source_code", fn.get("source", ""))
            results.append({
                "id": fid,
                "name": fn.get("name", fid),
                "file": fn.get("file_path", fn.get("file", "unknown")),
                "snippet": src[:300] + ("… [truncated]" if len(src) > 300 else ""),
                "rrf_score": round(rrf_score, 6),
            })
        return results

    def get_function_source(self, function_id: str) -> dict:
        fn = self.fn_map.get(function_id)
        if not fn:
            return {"error": f"Function ID '{function_id}' not found in index."}
        return {
            "id": fn["id"],
            "name": fn.get("name", "unknown"),
            "file": fn.get("file_path", fn.get("file", "unknown")),
            "source": fn.get("source_code", fn.get("source", "no source available")),
        }

    def get_graph_neighbors(self, function_id: str) -> dict:
        if not self.G.has_node(function_id):
            return {"error": f"Node '{function_id}' not found in dependency graph."}

        # get_neighborhood returns (all_ids_list, neighbour_ids_set)
        _, neighbour_ids = get_neighborhood(
            self.G, [function_id], max_depth=1, min_weight=MIN_WEIGHT
        )

        callees, callers = [], []
        for nb_id in neighbour_ids:
            if str(nb_id).startswith("__external__"):
                continue
            fn = self.fn_map.get(nb_id, {})
            entry = {
                "id": nb_id,
                "name": fn.get("name", nb_id),
                "file": fn.get("file_path", fn.get("file", "unknown")),
            }
            if self.G.has_edge(function_id, nb_id):
                w = self.G.get_edge_data(function_id, nb_id, {}).get("weight", 0)
                entry["weight"] = round(w, 3)
                callees.append(entry)
            else:
                w = self.G.get_edge_data(nb_id, function_id, {}).get("weight", 0)
                entry["weight"] = round(w, 3)
                callers.append(entry)

        return {
            "node": function_id,
            "callees": sorted(callees, key=lambda x: x["weight"], reverse=True),
            "callers": sorted(callers, key=lambda x: x["weight"], reverse=True),
        }

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": (
                "Search the codebase with hybrid BM25 + dense retrieval (RRF fusion). "
                "Returns the top matching functions with the short source code snippet. "
                "Use this first to find candidate functions for the query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":  {"type": "string",  "description": "The search query."},
                    "top_k":  {"type": "integer", "description": "Results to return (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_function_source",
            "description": (
                "Fetch the FULL source code of a specific function by ID. "
                "Call this after hybrid_search to read the complete implementation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "function_id": {"type": "string", "description": "Function ID from hybrid_search."},
                },
                "required": ["function_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_graph_neighbors",
            "description": (
                "Find callers and callees of a function in the dependency graph. "
                "Use this for multi-hop questions that require understanding relationships."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "function_id": {"type": "string", "description": "Function ID to look up."},
                },
                "required": ["function_id"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an expert code search agent with access to an indexed codebase.

Your goal: answer the user's query by iteratively searching and reading source code.

## Strategy
1. Start with `hybrid_search` using the user's query (rephrase if needed).
2. Call `get_function_source` on promising results to read the full implementation.
3. For questions about relationships between functions, use `get_graph_neighbors`.
4. You may search multiple times with different phrasings to find the best match.
5. Once you have enough context, write a clear final answer.

## Rules
- Base your answer ONLY on retrieved source code — never invent logic.
- If the answer cannot be found in the codebase, say so explicitly.
- Be concise. Reference specific function names and file paths.
"""

def run_agentic_rag(query: str, output_dir: Path) -> dict:
    """
    Runs the full agentic RAG pipeline for a single query.

    Returns:
        {
          "query": str,
          "retrieved_context": str,   # all code seen during the agent loop
          "generated_answer": str,   # final LLM answer
          "steps": int,
        }
    """
    tools = AgentTools(output_dir)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Query: {query}"},
    ]
    accumulated_context: list[dict[str, str]] = []
    steps = 0

    print(f"\n[Agentic RAG] Query: {query}")

    while steps < MAX_STEPS:
        MAX_RETRIES = 2
        response = None

        for attempt in range(MAX_RETRIES):
            try:
                response = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOL_SPECS,
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=1500,
                )
                break

            except Exception as e:
                err_msg = str(e)
                print(f"  [Step {steps+1}] LLM call failed (attempt {attempt+1}): {err_msg}")

                if "tool call validation failed" in err_msg:
                    feedback = (
                        "Your previous tool call had invalid argument types. "
                        "Ensure that:\n"
                        "- `top_k` must be an integer (e.g., 5, NOT \"5\")\n"
                        "- All arguments must match the JSON schema exactly.\n"
                        "Please try the tool call again with correct types."
                    )
                else:
                    feedback = (
                        f"The previous request failed with error: {err_msg}. "
                        "Please try again and fix the issue."
                    )

                messages.append({
                    "role": "user",
                    "content": feedback
                })

        if response is None:
            print("  LLM failed after retries. Skipping step.")
            break

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        messages.append(msg)

        if finish_reason == "stop" or not msg.tool_calls:
            print(f"  [Step {steps}] Agent finished.")
            break

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            print(f"  [Step {steps+1}] {fn_name}({fn_args})")

            try:
                if fn_name == "hybrid_search":
                    result = tools.hybrid_search(
                        fn_args.get("query", query),
                        fn_args.get("top_k", TOP_K),
                    )
                    for r in result:
                        accumulated_context.append({
                            "type": "hybrid_search",
                            "name": r['name'],
                            "fn_id": r['id'],
                            "file": r['file'],
                            "snippet": r['snippet']
                        })

                elif fn_name == "get_function_source":
                    result = tools.get_function_source(fn_args.get("function_id", ""))
                    if "source" in result:
                        accumulated_context.append({
                            "type": "full_source",
                            "name": result['name'],
                            "fn_id": result['id'],
                            "file": result['file'],
                            "source": result['source']
                        })

                elif fn_name == "get_graph_neighbors":
                    result = tools.get_graph_neighbors(fn_args.get("function_id", ""))
                    for nb in (result.get("callees", []) + result.get("callers", []))[:3]:
                        src = tools.get_function_source(nb["id"])
                        if "source" in src:
                            accumulated_context.append({
                                "type": "graph_neighbor",
                                "name": src['name'],
                                "fn_id": src['id'],
                                "file": src['file'],
                                "source": src['source']
                            })

                else:
                    result = {"error": f"Unknown tool: {fn_name}"}
            except Exception as e:
                result = {"error": f"Tool execution failed: {str(e)}"}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })
                feedback = (
                    "The previous tool call failed.\n"
                    f"Error: {str(e)}\n"
                    "Fix your arguments and call the tool again.\n"
                )
                messages.append({
                    "role": "user",
                    "content": feedback
                })
                continue

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, indent=2),
            })

        steps += 1
    
    # After the while loop, set a default
    generated_answer = ""
    error_flag = None

    last = messages[-1]
    last_role = last.role if hasattr(last, "role") else last.get("role")

    if response is None:
        error_flag = "LLM failed after all retries"

    elif last_role == "tool":
        try:
            final = groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=500,
            )
            generated_answer = final.choices[0].message.content or ""
        except Exception as e:
            error_flag = f"Final answer generation failed: {str(e)}"
    else:
        generated_answer = (
            last.content if hasattr(last, "content") else last.get("content", "")
        ) or ""
    
    print(f"  → Done in {steps} step(s). Context: {len(accumulated_context)} items.")

    return {
        "query": query,
        "retrieved_context": accumulated_context,
        "retrieved_context_str": _serialize_context(accumulated_context),
        "generated_answer": generated_answer,
        "error": error_flag,
        "steps": steps,
    }

def run_batch(
    output_dir: str,
    results_path: str = "agentic_rag_results.json",
    query_key: str = "query",
):
    """
    Runs agentic RAG on every query in a ground-truth JSON file.
    Saves incrementally so it can resume if interrupted.
    """
    
    num_results = 0

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    for idx, item in enumerate(results):
        if item.get("retrieved_context") is not None or item.get("generated_answer") is not None:
            num_results += 1
            print(f"  SKIP [{idx+1}/{len(results)}] — already has results")
            continue
        query = item.get(query_key, "")
        try:
            result = run_agentic_rag(query, Path(output_dir))
            if not result.get("error"):
                results[idx]["retrieved_context"] = result["retrieved_context_str"]
                results[idx]["generated_answer"] = result["generated_answer"]
                results[idx]["steps"] = result["steps"]
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4)
                num_results += 1
            else:
                print(f"  ERROR in query [{idx+1}/{len(results)}]: {result['error']}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. {num_results} results saved to {results_path}")
    return results

if __name__ == "__main__":
    run_batch(str(SAMPLE_REPO_OUTPUT_DIR), str(AGENTIC_RAG_OUTPUT_PATH_SINGLEHOP))
    run_batch(str(SAMPLE_REPO_OUTPUT_DIR), str(AGENTIC_RAG_OUTPUT_PATH_MULTIHOP))