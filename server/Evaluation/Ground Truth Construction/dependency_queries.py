import os
from dotenv import load_dotenv
import networkx as nx
import json
import time
import google.generativeai as genai

load_dotenv()  # Load environment variables from .env file

def extract_neighborhood_context(G: nx.DiGraph, root_node_id: str, max_neighbors: int = 3) -> str:
    """
    Given a root node ID, extracts its source code and the source code 
    of the functions it calls to build a multi-hop context payload.
    """
    if not G.has_node(root_node_id):
        return ""

    # 1. Grab the root node's data
    root_data = G.nodes[root_node_id]
    
    # We will build a list of node dictionaries to represent the context
    context_nodes = [{
        "role": "Primary Function",
        "name": root_data.get("name"),
        "file_path": root_data.get("file"),
        "source_code": root_data.get("source")
    }]

    # 2. Traverse outgoing edges (functions called by the root)
    neighbors = list(G.successors(root_node_id))
    
    # Filter out external library calls
    internal_neighbors = [n for n in neighbors if not str(n).startswith("__external__")]
    
    # SORTING LOGIC: Sort the remaining internal neighbors by edge weight in descending order
    # If an edge doesn't have a weight for some reason, default to 0.0
    internal_neighbors.sort(
        key=lambda n: G.get_edge_data(root_node_id, n).get("weight", 0.0), 
        reverse=True
    )
    
    # Now, when we take the top `max_neighbors`, we are guaranteed the highest confidence edges
    for neighbor_id in internal_neighbors[:max_neighbors]:
        n_data = G.nodes[neighbor_id]
        context_nodes.append({
            "role": f"Dependency (Weight: {G.get_edge_data(root_node_id, neighbor_id).get('weight', 0.0)})",
            "name": n_data.get("name"),
            "file_path": n_data.get("file"),
            "source_code": n_data.get("source"),
            "method": n_data.get("is_method", False)
        })

    # 3. Format the context into a clean, LLM-readable string
    payload = ""
    for node in context_nodes:
        payload += f"==========\n"
        payload += f"Role: {node['role']}\n"
        payload += f"File: {node['file_path']}\n"
        payload += f"Function/Class Name: {node['name']}\n"
        payload += f"Is Method: {node.get('method', False)}\n"
        payload += f"Code:\n{node['source_code']}\n"
        payload += f"==========\n\n"
        
    return payload

# Configure your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

llm_model = genai.GenerativeModel(
    'gemini-2.5-pro',
    generation_config={"response_mime_type": "application/json"}
)

def build_multi_hop_ground_truth(G: nx.DiGraph, output_file: str = "multihop_ground_truth.json"):
    
    # 1. Load existing progress and build a set of processed roots
    processed_roots = set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            ground_truth_dataset = json.load(f)
            
            # Extract root_ids that have already been successfully processed
            for item in ground_truth_dataset:
                if "root_id" in item:
                    processed_roots.add(item["root_id"])
                    
            print(f"Resuming: Loaded {len(ground_truth_dataset)} queries from disk.")
            print(f"Skipping {len(processed_roots)} already processed root nodes.")
    except (FileNotFoundError, json.JSONDecodeError):
        ground_truth_dataset = []
        print("Starting fresh dataset generation.")

    # 2. Get all candidate nodes
    candidate_roots = [
        n for n in G.nodes() 
        if not str(n).startswith("__external__") and len(list(G.successors(n))) > 0
    ]
    
    # 3. Filter out the ones we've already completed
    roots_to_process = [r for r in candidate_roots if r not in processed_roots]
    
    print(f"Total candidates: {len(candidate_roots)} | Remaining to process: {len(roots_to_process)}\n")

    for idx, root_id in enumerate(roots_to_process):
        print(f"[{idx+1}/{len(roots_to_process)}] Generating queries for: {G.nodes[root_id].get('name', root_id)}")
        
        neighborhood_payload = extract_neighborhood_context(G, root_id)
        
        prompt = f"""
        You are an expert software architect designing an evaluation dataset for a Code Search system.
        I am providing you with a "neighborhood" of connected functions from a codebase. 
        
        Your task is to generate 2 complex developer questions that STRICTLY REQUIRE synthesizing information from ALL the provided functions. 
        If a question can be answered by looking at just one function, it is a failure.
        
        Code Neighborhood:
        {neighborhood_payload}
        
        Output your response strictly as a JSON array:
        [
          {{
            "query": "<The complex multi-hop question>",
            "required_files": ["<file_path_1>", "<file_path_2>"]
          }}
        ]
        """
        
        try:
            response = llm_model.generate_content(prompt)
            response_json = json.loads(response.text)
            
            if isinstance(response_json, list):
                # INJECT ROOT ID: Tag each question with its origin node
                for item in response_json:
                    item["root_id"] = root_id
                
                ground_truth_dataset.extend(response_json)
                
                # INCREMENTAL SAVE: Write the updated list to disk immediately
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(ground_truth_dataset, f, indent=4)
                
                print(f"  -> Success! Total queries saved: {len(ground_truth_dataset)}")
                
            else:
                print(f"  -> Warning: Expected a JSON array, but got {type(response_json)}. Skipping.")
                
        except json.JSONDecodeError as e:
            print(f"  -> JSON Parsing Error: {e}")
        except Exception as e:
            print(f"  -> API Error for {root_id}: {e}")
        
        # Rate limit protection (Wait 4 seconds between calls)
        time.sleep(12)

    print("\nExtraction Complete! All data saved to", output_file)
    return ground_truth_dataset