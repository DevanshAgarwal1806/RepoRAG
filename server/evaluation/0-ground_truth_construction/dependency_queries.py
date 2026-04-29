import os
import sys
import json
import time
from pathlib import Path

import networkx as nx
from dotenv import load_dotenv
from openai import OpenAI

BASELINE_DIR = Path(__file__).resolve().parent
SERVER_DIR = BASELINE_DIR.parent.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
    
for candidate in [SERVER_DIR / ".env"]:
    if candidate.exists():
        load_dotenv(dotenv_path=candidate)
        break

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

from retriever.hybrid_retrieval_dependency import load_data
import tiktoken

# Llama models are close enough to cl100k
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def extract_neighborhood_context(
    G: nx.DiGraph, root_node_id: str, max_neighbors: int = 3
) -> str:
    """
    Given a root node ID, extracts its source code and the source code
    of the functions it calls to build a multi-hop context payload.
    """
    if not G.has_node(root_node_id):
        return None, 0, None

    # 1. Grab the root node's data
    root_data = G.nodes[root_node_id]

    context_nodes = [
        {
            "role": "Primary Function",
            "name": root_data.get("name"),
            "file_path": root_data.get("file"),
            "source_code": root_data.get("source"),
            "id": root_node_id
        }
    ]

    # 2. Traverse outgoing edges (functions called by the root)
    neighbors = list(G.successors(root_node_id))

    # Filter out external library calls
    internal_neighbors = [
        n for n in neighbors if not str(n).startswith("__external__") 
    ]

    # Sort by edge weight descending so highest-confidence edges come first
    internal_neighbors.sort(
        key=lambda n: G.get_edge_data(root_node_id, n).get("weight", 0.0),
        reverse=True,
    )
    
    if (len(internal_neighbors[:max_neighbors]) == 0):
        return None, 0, None

    for neighbor_id in internal_neighbors[:max_neighbors]:
        n_data = G.nodes[neighbor_id]
        weight = G.get_edge_data(root_node_id, neighbor_id).get("weight", 0.0)
        context_nodes.append(
            {
                "role": f"Dependency (Weight: {weight})",
                "name": n_data.get("name"),
                "file_path": n_data.get("file"),
                "source_code": n_data.get("source"),
                "method": n_data.get("is_method", False),
                "id": neighbor_id
            }
        )

    # 3. Format the context into a clean, LLM-readable string
    payload = ""
    for node in context_nodes:
        payload += "==========\n"
        payload += f"Role: {node['role']}\n"
        payload += f"File: {node['file_path']}\n"
        payload += f"Function ID: {node.get('id', 'N/A')}\n"
        payload += f"Function/Class Name: {node['name']}\n"
        payload += f"Is Method: {node.get('method', False)}\n"
        payload += f"Code:\n{node['source_code']}\n"
        payload += "==========\n\n"

    return payload, len(context_nodes), context_nodes


def build_multi_hop_ground_truth(
    G: nx.DiGraph,
    output_file: str = "multihop_ground_truth.json",
    max_neighbors: int = 3,  # FIX: expose max_neighbors as a parameter
) -> list:

    processed_roots: set = set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            ground_truth_dataset = json.load(f)

        for item in ground_truth_dataset:
            if "root_id" in item:
                processed_roots.add(item["root_id"])

        print(f"Resuming: Loaded {len(ground_truth_dataset)} queries from disk.")
        print(f"Skipping {len(processed_roots)} already processed root nodes.")
    except (FileNotFoundError, json.JSONDecodeError):
        ground_truth_dataset = []
        print("Starting fresh dataset generation.")

    candidate_roots = [
        n
        for n in G.nodes()
        if not str(n).startswith("__external__") and len(list(G.successors(n))) > 0
    ]

    roots_to_process = [r for r in candidate_roots if r not in processed_roots]

    print(
        f"Total candidates: {len(candidate_roots)} | "
        f"Remaining to process: {len(roots_to_process)}\n"
    )

    for idx, root_id in enumerate(roots_to_process):
        node_name = G.nodes[root_id].get("name", root_id)
        print(f"[{idx + 1}/{len(roots_to_process)}] Generating queries for: {node_name}")

        neighborhood_payload, num_context_nodes, context_nodes = extract_neighborhood_context(
            G, root_id, max_neighbors=max_neighbors
        )
        
        if neighborhood_payload is None:
            print(f"  -> Skipping '{node_name}' due to lack of valid neighbors.")
            continue

        prompt = f"""
        You are an expert software architect designing an evaluation dataset for a Code Search system.
        I am providing you with a "neighborhood" of connected functions from a codebase.

        Your task is to generate 2 complex developer questions that STRICTLY REQUIRE synthesizing
        information from ALL the provided functions.
        If a question can be answered by looking at just one function, it is a failure.

        Code Neighborhood:
        {neighborhood_payload}

        Output your response strictly as a JSON object with a single key named "queries".
        The value must be an array of question objects. Each object in "required_chunks" must include
        the fully qualified function name and its file path exactly as shown in the code neighborhood above.
        {{
        "queries": [
            {{
            "query": "<The complex multi-hop question>",
            "required_chunks": [
                {{
                "function_name": "<Exact_Function_Name_1>",
                "file_path": "<Exact_File_Path_1>"
                }},
                {{
                "function_name": "<Exact_Function_Name_2>",
                "file_path": "<Exact_File_Path_2>"
                }}
            ]
            }}
        ]
        }}
        """

        max_retries = 3
        retry_delay = 60
        success = False
        
        print(f"  -> Prompt token count: {count_tokens(prompt)}")
        print(f"  -> Number of context nodes provided: {num_context_nodes}")

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a senior software architect. "
                                "Output strictly valid JSON. Do not wrap in markdown."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )

                response_text = (
                    response.choices[0]
                    .message.content
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )
                response_json = json.loads(response_text)
                extracted_array = response_json.get("queries")

                if not isinstance(extracted_array, list):
                    print(
                        f"  -> Warning: 'queries' key missing or not a list "
                        f"(Attempt {attempt + 1}/{max_retries}). Retrying..."
                    )
                    continue

                for item in extracted_array:
                    item["root_id"] = root_id
                    if "required_chunks" in item and isinstance(item["required_chunks"], list):
                        seen = set()
                        unique_chunks = []
                        for chunk in item["required_chunks"]:
                            identifier = (chunk.get("function_name"), chunk.get("file_path"))
                            if identifier not in seen:
                                seen.add(identifier)
                                unique_chunks.append(chunk)
                        item["required_chunks"] = unique_chunks
                    
                to_add = []
                for idx, query in enumerate(extracted_array):
                    for chunk in query.get("required_chunks", []):
                        for nodes in context_nodes:
                            if nodes["file_path"] == chunk["file_path"] and nodes["name"] == chunk["function_name"]:
                                query.setdefault("required_functions", []).append(
                                    nodes.get("id", "N/A")
                                )
                                break
                    query.pop("required_chunks", None)  # Remove the original required_chunks key
                    if len(query.get("required_functions", [])) != 0:
                        to_add.append(idx)
                
                ground_truth_dataset.extend(extracted_array)

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(ground_truth_dataset, f, indent=4)

                print(f"  -> Success! Total queries saved: {len(ground_truth_dataset)}")
                success = True
                break

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    print(
                        f"  -> TPM Limit Hit (Attempt {attempt + 1}/{max_retries}). "
                        f"Sleeping for {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    print(f"  -> API/Parse Error: {e}")
                    break

        if not success:
            print(f"  -> Skipped '{node_name}' after {max_retries} attempts.")

        time.sleep(6)

    print("\nExtraction complete! All data saved to", output_file)
    return ground_truth_dataset

if __name__ == "__main__":
    G, function_map = load_data(str(SERVER_DIR / "sample_repository_output"))
    build_multi_hop_ground_truth(G, output_file=str(BASELINE_DIR / "multihop_ground_truth.json"))