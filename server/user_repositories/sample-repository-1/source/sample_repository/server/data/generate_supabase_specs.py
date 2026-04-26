import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL_AGENTIC_AI")
SUPABASE_KEY = os.getenv("SUPABASE_KEY_AGENTIC_AI")

SCHEMAS_TO_FETCH = ["public", "employee_assignment_data"] 

def download_all_specs():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: Missing credentials in .env")
        return

    save_dir = os.path.join("data", "openapi_specs")
    os.makedirs(save_dir, exist_ok=True)

    for schema in SCHEMAS_TO_FETCH:
        print(f"Fetching spec for schema: {schema}...")
        
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Accept-Profile": schema,  # Tells Supabase which schema to use for data
            "Content-Profile": schema  # Tells Supabase which schema to DOCUMENT in the spec
        }
        
        try:
            # We hit the base rest/v1/ endpoint with the specific schema headers
            response = requests.get(f"{SUPABASE_URL}/rest/v1/", headers=headers)
            response.raise_for_status()
            
            file_path = os.path.join(save_dir, f"supabase_{schema}_spec.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(response.json(), f, indent=2)
                
            print(f"Saved: {file_path}")
            
        except Exception as e:
            print(f"Failed to fetch {schema}: {e}")

if __name__ == "__main__":
    download_all_specs()