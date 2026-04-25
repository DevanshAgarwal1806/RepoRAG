import sys
from pathlib import Path
from urllib import response
import networkx as nx
import os
import json
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

import tiktoken

# Llama models are close enough to cl100k
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

BASELINE_DIR = Path(__file__).resolve().parent
SERVER_DIR = BASELINE_DIR.parent.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

for candidate in [SERVER_DIR / ".env"]:
    if candidate.exists():
        load_dotenv(dotenv_path=candidate)
        break

from retriever.graph_context import load_data

from openai import OpenAI

# Initialize the Client using Groq
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

def extract_single_node_context(G: nx.DiGraph, node_id: str) -> str:
    """
    Given a node ID, extracts just its own source code and metadata.
    """
    if not G.has_node(node_id):
        return ""

    node_data = G.nodes[node_id]
    
    payload = f"==========\n"
    payload += f"File: {node_data.get('file')}\n"
    payload += f"Function/Class Name: {node_data.get('name')}\n"
    payload += f"Is Method: {node_data.get('is_method', False)}\n"
    payload += f"Code:\n{node_data.get('source')}\n"
    payload += f"==========\n"
        
    return payload

def build_single_hop_ground_truth(G: nx.DiGraph, output_file: str = "singlehop_ground_truth.json"):
    
    # 1. Load existing progress and build a set of processed nodes
    processed_nodes = set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            ground_truth_dataset = json.load(f)
            
            # Extract node_ids that have already been successfully processed
            for item in ground_truth_dataset:
                if "node_id" in item:
                    processed_nodes.add(item["node_id"])
                    
            print(f"Resuming: Loaded {len(ground_truth_dataset)} queries from disk.")
            print(f"Skipping {len(processed_nodes)} already processed nodes.")
    except (FileNotFoundError, json.JSONDecodeError):
        ground_truth_dataset = []
        print("Starting fresh single-hop dataset generation.")

    # 2. Get all candidate nodes (internal code only)
    candidate_nodes = [
        n for n in G.nodes() 
        if not str(n).startswith("__external__")
    ]
    
    # Optional: Filter out extremely short nodes (like 1-liners) that don't make good questions
    candidate_nodes = [n for n in candidate_nodes if len(G.nodes[n].get("source", "")) > 30]

    # 3. Filter out the ones we've already completed
    nodes_to_process = [n for n in candidate_nodes if n not in processed_nodes]
    
    print(f"Total candidates: {len(candidate_nodes)} | Remaining to process: {len(nodes_to_process)}\n")

    for idx, node_id in enumerate(nodes_to_process):
        print(f"[{idx+1}/{len(nodes_to_process)}] Generating 1:1 queries for: {G.nodes[node_id].get('name', node_id)}")
        
        single_payload = extract_single_node_context(G, node_id)
        
        prompt = f"""
        You are an expert software engineer reviewing a codebase. I am providing you with a single function/code snippet.

        Your task is to generate 3 realistic developer questions where THIS EXACT snippet is the perfect and only answer. 

        Generate three distinct types of questions:
        1. "exact": A question that explicitly mentions a function name, class, or specific variable found in the code.
        2. "semantic": A high-level question about the purpose or logic of the code without using the exact variable/function names.
        3. "structural": A question asking about what this specific component is responsible for within the wider system.

        Code Snippet:
        {single_payload}

        Output your response strictly as a JSON array of objects. Do not include markdown formatting or extra text.
        {{
            "queries": [
                {{
                    "query_type": "exact",
                    "query": "<your exact match question>",
                    "target_file": "<file_path>"
                }},
                {{
                    "query_type": "semantic",
                    "query": "<your semantic question>",
                    "target_file": "<file_path>"
                }},
                {{
                    "query_type": "structural",
                    "query": "<your structural question>",
                    "target_file": "<file_path>"
                }}
            ]
        }}
        """
        
        max_retries = 3
        retry_delay = 60 # Groq limits reset every minute
        print(f"  -> Prompt token count: {count_tokens(prompt)}")
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile", # The 70B Reasoning Engine
                    messages=[
                        {"role": "system", "content": "You are a senior software engineer. Output strictly valid JSON arrays. Do not wrap in markdown."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"} 
                )
                
                # 1. Parse the text into a Python dictionary
                response_text = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
                response_json = json.loads(response_text)
                
                # 2. Extract the array using the exact key we prompted for
                extracted_array = response_json.get("queries")
                
                # 3. Process the array
                if isinstance(extracted_array, list):
                    for item in extracted_array:
                        item["node_id"] = node_id
                    
                    ground_truth_dataset.extend(extracted_array)
                    
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(ground_truth_dataset, f, indent=4)
                        
                    print(f"  -> Success! Total 1:1 queries saved: {len(ground_truth_dataset)}")
                    break 
                else:
                    print("  -> Warning: The model did not return a 'queries' array.")
                    break
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    print(f"  -> TPM Limit Hit (Attempt {attempt + 1}/{max_retries}). Sleeping for 60s...")
                    time.sleep(retry_delay)
                else:
                    print(f"  -> API/Parse Error: {e}")
                    break
        
        # The mathematically safe sleep for 12K TPM
        time.sleep(6)
        
    return ground_truth_dataset

if __name__ == "__main__":
    G, function_map = load_data(str(SERVER_DIR / "sample_repository_output"))
    
    build_single_hop_ground_truth(G, output_file=str(BASELINE_DIR / "singlehop_ground_truth.json"))