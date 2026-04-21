import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import argparse

def generate_rag_answer(output_dir: str) -> str:
    current_dir = Path(__file__).resolve().parent
    server_dir = current_dir.parent  
    env_path = server_dir / ".env"
    
    output_dir = Path(output_dir)
    payload_path = output_dir / "final_llm_payload.md"
    
    # 2. Load Environment Variables
    load_dotenv(dotenv_path=env_path)
    
    api_key = os.getenv("ANSWER_GENERATION_LLM_KEY") 
    
    if not api_key or not api_key.startswith("gsk_"):
        raise ValueError("Valid Groq API Key (starting with 'gsk_') not found in .env file.")

    # 3. Read the Prepared Context & Query
    if not payload_path.exists():
        raise FileNotFoundError(f"Could not find the LLM payload at {payload_path}.")
        
    with open(payload_path, "r", encoding="utf-8") as f:
        llm_payload = f.read()

    # 4. Initialize the Groq Client
    client = Groq(api_key=api_key)

    print("--- Sending Context to Groq LLM ---")
    
    # 5. Generate the Answer
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Groq's fast Llama 3 70B model
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert senior software engineer. "
                        "You will be provided with a user query and a heavily filtered context from a codebase. "
                        "The context includes specific functions and their neighboring dependency graph. "
                        "Answer the user's question accurately using ONLY the provided codebase context. "
                        "If the answer is not contained within the context, explicitly state that you cannot answer based on the provided code."
                    )
                },
                {
                    "role": "user",
                    "content": llm_payload
                }
            ],
            temperature=0.2, 
            max_tokens=500
        )
        
        final_answer = response.choices[0].message.content
        
        print("\n--- Final Generated Answer ---")
        print(final_answer)
        
        return final_answer

    except Exception as e:
        print(f"An error occurred during generation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Generator ")
    parser.add_argument("--output", "-o", required=True, help="Output directory for results")
    args = parser.parse_args()
    generate_rag_answer(args.output)