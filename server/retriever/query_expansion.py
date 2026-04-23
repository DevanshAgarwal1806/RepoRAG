import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def get_client() -> OpenAI | None:
    api_key = os.environ.get("QUERY_EXPANSION_LLM_KEY")
    if not api_key:
        return None

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

def expand_query(user_query: str) -> str:
    """
    Takes a natural language user query and uses an LLM to generate 
    software engineering synonyms returned in a strict JSON format.
    """
    
    system_prompt = (
        "You are an expert software engineering assistant. "
        "The user will provide a short natural language query about a codebase. "
        "Expand this query by providing 5 to 10 highly relevant technical synonyms, "
        "standard library names, or programming concepts related to the query. "
        "You MUST respond ONLY with a valid JSON object. "
        "The JSON must have a single key 'synonyms' mapping to an array of strings. "
        "Example: {\"synonyms\": [\"SQL\", \"ORM\", \"psycopg2\"]}"
    )

    try:
        client = get_client()
        if client is None:
            return user_query

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {user_query}"}
            ],
            temperature=0.3, 
            max_tokens=100,
            # 2. Force the model to output a JSON object
            response_format={"type": "json_object"} 
        )
        
        # Extract the raw JSON string
        raw_json_string = response.choices[0].message.content.strip()
        
        # 3. Parse the JSON safely
        parsed_data = json.loads(raw_json_string)
        
        # Extract the list of synonyms (default to empty list if key is missing)
        synonyms_list = parsed_data.get("synonyms", [])
        
        # Convert the list back into a space-separated string for BM25
        synonyms_string = " ".join(synonyms_list)
        
        # Combine with original query
        expanded_query = f"{user_query} {synonyms_string}"
        
        return expanded_query

    except json.JSONDecodeError:
        print("Error: The LLM did not return valid JSON.")
        return user_query
    except Exception as e:
        print(f"Error during query expansion: {e}")
        return user_query

if __name__ == "__main__":
    test_queries = [
        "how do I handle user login",
        "fetch data from the database",
        "make an API request to the external server"
    ]

    print("--- Testing JSON-based Query Expansion ---")
    for q in test_queries:
        expanded = expand_query(q)
        print(f"Original : {q}")
        print(f"Expanded : {expanded}")
