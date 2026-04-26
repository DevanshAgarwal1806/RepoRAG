import os
from pathlib import Path
import sys
from dotenv import load_dotenv
import argparse

RETRIEVER_DIR = Path(__file__).resolve().parent
SERVER_DIR = RETRIEVER_DIR.parent

TOKENS_ANSWER = 500

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))


def estimate_context_size(payload: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        estimated_input = len(enc.encode(payload))
    except Exception:
        estimated_input = max(1, len(payload) // 4)

    return estimated_input + TOKENS_ANSWER + 256


def generate_rag_answer(
    output_dir: str, 
    llm_payload: str = None, 
    provider: str = "groq", 
    model_name: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Generates an answer using either the Groq API or a local Ollama instance.
    
    Args:
        output_dir: Directory containing the context payload.
        llm_payload: The actual text payload (optional, will read from file if None).
        provider: "groq" or "ollama".
        model_name: The specific model ID to use for the selected provider.
    """
    current_dir = Path(__file__).resolve().parent
    server_dir = current_dir.parent  
    env_path = server_dir / ".env"
    
    output_dir = Path(output_dir)

    # 1. Read the Prepared Context & Query
    if llm_payload is None:
        payload_path = output_dir / "final_llm_payload.md"
        if not payload_path.exists():
            raise FileNotFoundError(f"Could not find the LLM payload at {payload_path}.")
            
        with open(payload_path, "r", encoding="utf-8") as f:
            llm_payload = f.read()

    # 2. Define the Shared System Prompt
    system_prompt = (
        "You are an expert senior software engineer. "
        "You will be provided with a user query and a heavily filtered context from a codebase. "
        "The context includes specific functions and their neighboring dependency graph. "
        "Answer the user's question accurately using ONLY the provided codebase context. "
        "If the answer is not contained within the context, explicitly state that you cannot answer based on the provided code."
    )

    if provider == "ollama":
        try:
            import ollama

            safe_ctx = estimate_context_size(llm_payload)
            print(f"Generating local answer using Ollama ({model_name})...")
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": llm_payload}
                ],
                options={
                    "num_ctx": safe_ctx, # Bumped to 10K to ensure plenty of room for output
                    "num_predict": TOKENS_ANSWER,
                    "temperature": 0.1
                }
            )
            return response['message']['content']
        except Exception as e:
            print(f"An error occurred during local Ollama generation: {e}")
            return ""

    elif provider == "groq":
        load_dotenv(dotenv_path=env_path)
        api_key = os.getenv("ANSWER_GENERATION_LLM_KEY") 
        
        if not api_key or not api_key.startswith("gsk_"):
            raise ValueError("Valid Groq API Key (starting with 'gsk_') not found in server/.env.")

        from groq import Groq

        client = Groq(api_key=api_key)
        
        try:
            print(f"Generating remote answer using Groq ({model_name})...")
            response = client.chat.completions.create(
                model=model_name, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": llm_payload}
                ],
                temperature=0.1,
                max_tokens=TOKENS_ANSWER
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred during Groq generation: {e}")
            return ""
            
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'groq' or 'ollama'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RepoRAG LLM Generator")
    parser.add_argument("--output", "-o", required=True, help="Output directory for results")
    parser.add_argument("--provider", "-p", choices=["groq", "ollama"], default="groq", help="Which LLM backend to use")
    parser.add_argument("--model", "-m", default="llama-3.3-70b-versatile", help="The model tag (e.g., qwen2.5-coder:7b)")
    
    args = parser.parse_args()
    
    # Run the generator and print the output
    final_answer = generate_rag_answer(args.output, provider=args.provider, model_name=args.model)
    
    print("\n--- Final Generated Answer ---\n")
    print(final_answer)
