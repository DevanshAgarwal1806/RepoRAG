import os
import re
import json
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables (e.g., SUPABASE_URL) from a .env file
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
# Point this to the specific directory containing your CSVs and JSON
DATA_DIR = "data/custom_data/Employee_Assignment_Data" 

# Your Supabase Connection String
SUPABASE_URL = os.getenv("SUPABASE_CONN_AGENTIC_AI")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def sanitize_identifier(name: str) -> str:
    """
    Cleans directory, file, and column names to be valid PostgreSQL identifiers.
    - Removes file extensions.
    - Replaces spaces and special characters with underscores.
    - Converts everything to lowercase.
    """
    name = os.path.splitext(name)[0]
    clean_name = re.sub(r'\W+', '_', name)
    return clean_name.lower().strip('_')

def get_metadata(directory: str) -> dict:
    """
    Scans the target directory for a .json file and extracts the metadata.
    This metadata is used to apply PostgreSQL Comments to the schema and tables.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not read JSON metadata in {filename}: {e}")
                return {}
    return {}

def run_ingestion_pipeline():
    if not SUPABASE_URL:
        print("Error: SUPABASE_URL environment variable is missing.")
        return

    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found.")
        return

    # 1. Determine Schema Name dynamically from the Directory Name
    raw_dir_name = os.path.basename(os.path.normpath(DATA_DIR))
    schema_name = sanitize_identifier(raw_dir_name)
    print(f"Starting data pipeline for Schema: '{schema_name}'")

    # 2. Extract descriptions from the JSON file
    metadata = get_metadata(DATA_DIR)
    schema_description = metadata.get("description", f"Auto-generated schema from {raw_dir_name} directory.")
    table_descriptions = metadata.get("tables", {})

    # 3. Connect to Supabase via SQLAlchemy
    engine = create_engine(SUPABASE_URL)

    # 4. Manage the Schema (Drop if exists, Create fresh, Apply Description)
    with engine.begin() as conn: # engine.begin() automatically manages transactions/commits
        try:
            print(f"Dropping schema '{schema_name}' (and all its tables) if it exists...")
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;"))
            
            print(f"Creating fresh schema '{schema_name}'...")
            conn.execute(text(f"CREATE SCHEMA {schema_name};"))
            
            # Apply the Schema Description from JSON
            # We use parameterized queries (:desc) to prevent SQL injection and syntax errors
            comment_sql = text(f"COMMENT ON SCHEMA {schema_name} IS :desc;")
            conn.execute(comment_sql, {"desc": schema_description})
            print("Schema description successfully applied.")
            
        except Exception as e:
            print(f"Schema creation failed: {e}")
            return

    # 5. Iterate through CSV files and build tables inside the new schema
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".csv"):
            file_path = os.path.join(DATA_DIR, filename)
            table_name = sanitize_identifier(filename)
            
            print(f"\nProcessing '{filename}' -> Target: '{schema_name}.{table_name}'")
            
            try:
                # Read CSV into Pandas DataFrame
                df = pd.read_csv(file_path)
                
                # Sanitize all column names to prevent SQL syntax errors
                df.columns = [sanitize_identifier(col) for col in df.columns]
                
                # Push the DataFrame to Supabase
                # 'schema=schema_name' ensures it goes into our grouped folder, not 'public'
                df.to_sql(name=table_name, con=engine, schema=schema_name, if_exists="replace", index=False)
                print(f"Inserted {len(df)} rows into '{schema_name}.{table_name}'.")
                
                # 6. Apply Table Descriptions if they exist in the JSON
                # This must be done AFTER to_sql because the table must exist first
                if table_name in table_descriptions:
                    with engine.begin() as conn:
                        table_desc = table_descriptions[table_name]
                        conn.execute(
                            text(f"COMMENT ON TABLE {schema_name}.{table_name} IS :tdesc;"), 
                            {"tdesc": table_desc}
                        )
                        print(f"   ↳ Applied JSON description to table '{table_name}'.")
                        
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    print(f"\nPipeline Complete. All CSV files in '{DATA_DIR}' have been ingested into the '{schema_name}' schema on Supabase.")

if __name__ == "__main__":
    run_ingestion_pipeline()