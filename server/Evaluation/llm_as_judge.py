import json
import time
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Configure your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use Gemini 2.5 Pro for deep reasoning
judge_model = genai.GenerativeModel(
    'gemini-2.5-pro',
    generation_config={"response_mime_type": "application/json"}
)

def evaluate_pipeline_results(results_filepath: str, evaluation_type: str = "relevance"):
    """
    Reads pipeline results, calls the LLM Judge, and appends scores.
    evaluation_type can be "relevance" (for retrieval) or "faithfulness" (for generation).
    """
    
    # 1. Load the results of your pipeline
    with open(results_filepath, "r", encoding="utf-8") as f:
        pipeline_results = json.load(f)

    print(f"Starting {evaluation_type.upper()} evaluation for {len(pipeline_results)} items...")

    for idx, item in enumerate(pipeline_results):
        # Skip if already judged (allows resuming)
        if f"{evaluation_type}_score" in item:
            continue
            
        print(f"[{idx+1}/{len(pipeline_results)}] Judging Query: {item['query'][:50]}...")

        # 2. Select the correct rubric
        if evaluation_type == "relevance":
            prompt = f"""
            You are an expert code reviewer evaluating a Code Search system.
            Your task is to evaluate if the "Retrieved Code" contains the necessary information to completely answer the "User Query".
            
            User Query: {item['query']}
            Retrieved Code:
            {item['retrieved_context']}
            
            Evaluate the context using the following 0-2 scale:
            0 (Irrelevant): The code does not contain any logic related to the query.
            1 (Partial): The code contains some relevant variables or base logic, but is missing the core implementation needed to answer the query fully.
            2 (Perfect): The code contains the exact functions and logic required to perfectly answer the query.

            Output your evaluation STRICTLY in JSON format:
            {{
            "score": <int>,
            "reasoning": "<One concise sentence explaining the score>"
            }}
            
            """
        elif evaluation_type == "faithfulness":
            prompt = f"""
            You are an expert code reviewer evaluating an AI assistant.
            Your task is to evaluate if the "Generated Answer" is faithful to the "Retrieved Code". The AI must NOT invent logic, functions, or libraries that do not exist in the retrieved code.

            User Query: {item['query']}
            Retrieved Code:
            {item['retrieved_context']}

            Generated Answer:
            {item['generated_answer']}

            Evaluate the faithfulness using the following 0-2 scale:
            0 (Hallucination): The answer relies on logic, functions, or facts NOT present in the retrieved code.
            1 (Unverifiable/Vague): The answer is too vague to verify against the code, or ignores the code entirely.
            2 (Faithful): The answer is directly supported by the retrieved code and clearly references it.

            Output your evaluation STRICTLY in JSON format:
            {{
            "score": <int>,
            "reasoning": "<One concise sentence explaining the score>"
            }}
            """

        # 3. Call the LLM Judge
        try:
            response = judge_model.generate_content(prompt)
            judge_output = json.loads(response.text)
            
            # 4. Save the score and reasoning back to the dictionary
            item[f"{evaluation_type}_score"] = judge_output.get("score", 0)
            item[f"{evaluation_type}_reasoning"] = judge_output.get("reasoning", "Parse error")
            
            # Incremental Save to disk
            with open(results_filepath, "w", encoding="utf-8") as f:
                json.dump(pipeline_results, f, indent=4)
                
            print(f"  -> Score: {judge_output.get('score')} | Reasoning: {judge_output.get('reasoning')}")

        except Exception as e:
            print(f"  -> API/Parse Error: {e}")

        # Rate Limit Protection (5 RPM)
        time.sleep(12)

    print("\nEvaluation Complete!")
    return pipeline_results

def calculate_average_scores(results_filepath):
    with open(results_filepath, "r", encoding="utf-8") as f:
        results = json.load(f)
        
    total_relevance = sum(item.get("relevance_score", 0) for item in results)
    total_faithfulness = sum(item.get("faithfulness_score", 0) for item in results)
    
    avg_relevance = total_relevance / len(results)
    avg_faithfulness = total_faithfulness / len(results)
    
    print("--- Final Ablation Study Results ---")
    print(f"Average Context Relevance (0-2): {avg_relevance:.2f}")
    print(f"Average Answer Faithfulness (0-2): {avg_faithfulness:.2f}")