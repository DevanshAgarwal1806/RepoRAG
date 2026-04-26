import os
import json
import yaml
from langchain_community.agent_toolkits.openapi.toolkit import OpenAPIToolkit
from langchain_community.utilities.requests import RequestsWrapper
from langchain_community.tools.json.tool import JsonSpec
from langchain_groq import ChatGroq

def get_openapi_tools():
    """
    Reads ALL OpenAPI/Swagger specs from data/openapi_specs/ 
    and dynamically generates a combined list of tools using Groq.
    """
    specs_dir = os.path.join(os.getcwd(), "data", "openapi_specs")
    
    if not os.path.exists(specs_dir):
        return []

    spec_files = [f for f in os.listdir(specs_dir) if f.endswith(('.json', '.yaml', '.yml'))]
    
    if not spec_files:
        print("No OpenAPI specs found in data/openapi_specs/. Skipping.")
        return []

    # Initialize a master list to hold tools from all files
    all_tools = []

    # Initialize the Groq LLM once to reuse for all toolkits
    llm = ChatGroq(
        model="llama3-70b-8192", 
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY_OPENAPI")
    )
    
    # Standard wrapper for making the network calls
    requests_wrapper = RequestsWrapper()

    for file in spec_files:
        filepath = os.path.join(specs_dir, file)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                if file.endswith(".json"):
                    raw_spec = json.load(f)
                else:
                    raw_spec = yaml.safe_load(f)
                    
            json_spec = JsonSpec(dict_=raw_spec, max_value_length=4000)
            
            # Create a toolkit for this specific specification
            toolkit = OpenAPIToolkit.from_llm(
                llm=llm,
                json_spec=json_spec,
                requests_wrapper=requests_wrapper,
                allow_dangerous_requests=True
            )
            
            # Extend the master list with the tools from this toolkit
            new_tools = toolkit.get_tools()
            all_tools.extend(new_tools)
            
            print(f"Dynamically generated {len(new_tools)} tools from: {file}")
            
        except Exception as e:
            print(f"Failed to parse OpenAPI spec {file}: {e}")
            continue # Move to the next file if one fails

    print(f"Total tools generated across all specs: {len(all_tools)}")
    return all_tools