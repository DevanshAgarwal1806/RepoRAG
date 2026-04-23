import networkx as nx
import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Configure your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Updated to the current stable Pro model
llm_model = genai.GenerativeModel(
    'gemini-2.5-pro', 
    generation_config={"response_mime_type": "application/json"}
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
        [
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
        """
        
        try:
            response = llm_model.generate_content(prompt)
            response_json = json.loads(response.text)
            
            if isinstance(response_json, list):
                # INJECT NODE ID: Tag each question with its exact AST node origin
                for item in response_json:
                    item["node_id"] = node_id
                
                ground_truth_dataset.extend(response_json)
                
                # INCREMENTAL SAVE
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(ground_truth_dataset, f, indent=4)
                
                print(f"  -> Success! Total 1:1 queries saved: {len(ground_truth_dataset)}")
                
            else:
                print(f"  -> Warning: Expected a JSON array, but got {type(response_json)}. Skipping.")
                
        except json.JSONDecodeError as e:
            print(f"  -> JSON Parsing Error: {e}")
        except Exception as e:
            print(f"  -> API Error for {node_id}: {e}")
        
        # 12 seconds ensures you stay under the 5 Requests-Per-Minute free tier limit
        time.sleep(12) 

    print("\nExtraction Complete! All single-hop data saved to", output_file)
    return ground_truth_dataset