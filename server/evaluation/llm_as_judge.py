import json
import time
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parent
SERVER_DIR = EVALUATION_DIR.parent
GROUND_TRUTH_DIR = EVALUATION_DIR / "ground_truth_construction"
DEFAULT_OUTPUT_DIR = SERVER_DIR / "sample_repository_output"
COMPLETE_SYSTEM_EVALUATION_DIR = EVALUATION_DIR / "complete_system_evaluation"
AGENTIC_SYSTEM_OUTPUT_DIR = EVALUATION_DIR / "agentic_ai_baseline"

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

JUDGE_MODEL = "gemma-4-31b-it"


def _call_judge(prompt: str) -> dict:
    """Calls Gemma 4 31B and returns parsed JSON. Raises on failure."""
    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=512,
        ),
    )
    output = json.loads(response.text)

    winner = output.get("winner", "").strip().upper()
    if winner not in ("A", "B", "TIE"):
        raise ValueError(f"Unexpected winner value: '{winner}'")

    output["winner"] = winner
    return output


def judge_context_relevance(query: str, rag_context: str, agentic_context: str) -> dict:
    """
    Head-to-head: which system retrieved more relevant code snippets for the query?
    A = RAG system, B = Agentic system
    """
    prompt = f"""You are an expert code search evaluator. Your task is to judge which retrieved code context is MORE RELEVANT and USEFUL for answering the given user query.

    User Query:
    {query}

    --- Context A ---
    {rag_context}

    --- Context B ---
    {agentic_context}

    Evaluation Criteria:
    1. Relevance     — Does the context directly contain code related to the query?
    2. Completeness  — Does it capture the full logic needed (not just fragments)?
    3. Precision     — Is the retrieved code focused, or does it contain a lot of noise/unrelated code?

    Analyze Context A and Context B strictly against these criteria.
    Output exactly one of: "A", "B", or "Tie".

    Respond ONLY in this JSON format with no extra text:
    {{
    "winner": "<A | B | Tie>",
    "reasoning": "<One concise sentence explaining your decision>"
    }}
    """
    return _call_judge(prompt)


def judge_answer_quality(
    query: str,
    rag_context: str, rag_answer: str,
    agentic_context: str, agentic_answer: str
) -> dict:
    """
    Head-to-head: which system generated a better answer, grounded in its own retrieved context?
    A = RAG system, B = Agentic system
    """
    prompt = f"""You are an expert code reviewer. Your task is to judge which AI system produced a BETTER answer to the user query.
    Each system has its own retrieved code context and generated answer. Judge each answer against its own context — penalise any system that hallucinates logic not present in its retrieved code.

    User Query:
    {query}

    === System A ===
    Retrieved Context:
    {rag_context}

    Generated Answer:
    {rag_answer}

    === System B ===
    Retrieved Context:
    {agentic_context}

    Generated Answer:
    {agentic_answer}

    Evaluation Criteria (in order of priority):
    1. Faithfulness   — Is the answer grounded in its own retrieved context? Does it hallucinate?
    2. Correctness    — Is the answer technically accurate?
    3. Completeness   — Does it fully address the query?
    4. Clarity        — Is the explanation clear and easy to follow?
    5. Code Quality   — Is any code snippet clean, idiomatic, and functional?

    Analyze Answer A and Answer B strictly against these criteria.
    Output exactly one of: "A", "B", or "Tie".

    Respond ONLY in this JSON format with no extra text:
    {{
    "winner": "<A | B | Tie>",
    "reasoning": "<One concise sentence explaining your decision>"
    }}
    """
    return _call_judge(prompt)


def run_comparison(
    comparison_results_filepath: str,
    agentic_results_filepath: str,
    rag_results_filepath: str
) -> list:
    """
    Main runner. Loads RAG and Agentic results, runs both judges on each item,
    and saves incrementally to allow resuming if interrupted.
    """
    comparison_path = EVALUATION_DIR / comparison_results_filepath

    with open(AGENTIC_SYSTEM_OUTPUT_DIR / agentic_results_filepath, "r", encoding="utf-8") as f:
        agentic_results = json.load(f)

    with open(COMPLETE_SYSTEM_EVALUATION_DIR / rag_results_filepath, "r", encoding="utf-8") as f:
        rag_results = json.load(f)

    total = len(rag_results)

    if comparison_path.exists():
        with open(comparison_path, "r", encoding="utf-8") as f:
            comparison_results = json.load(f)
    else:
        comparison_results = []

    while len(comparison_results) < total:
        comparison_results.append({})

    print(f"Starting RAG vs Agentic comparison for {total} items using {JUDGE_MODEL}...\n")

    for idx, (rag, agentic) in enumerate(zip(rag_results, agentic_results)):
        query = rag.get("query", "")
        print(f"[{idx+1}/{total}] Query: {query[:70]}...")

        comparison_results[idx]["query"] = query
        
        if comparison_results[idx].get("context_winner") in (None, "ERROR"):
            try:
                result = judge_context_relevance(
                    query=query,
                    rag_context=rag.get("retrieved_context", ""),
                    agentic_context=agentic.get("retrieved_context_str", ""),
                )
                comparison_results[idx].update({
                    "context_winner": result["winner"],
                    "context_reasoning": result.get("reasoning", ""),
                })
                print(f"[Context] Winner: {result['winner']} | {result.get('reasoning', '')}")
            except Exception as e:
                print(f"[Context] ERROR: {e}")
                # FIX 5: Save error state so it appears in output and is retried next run
                comparison_results[idx].update({
                    "context_winner": "ERROR",
                    "context_reasoning": str(e),
                })

            time.sleep(3)
        else:
            print(f"[Context] Already judged → {comparison_results[idx]['context_winner']}. Skipping.")

        # ── Round 2: Answer Quality ────────────────────────────────────────────
        # FIX 4: Same retry logic for answer round
        if comparison_results[idx].get("answer_winner") in (None, "ERROR"):
            try:
                result = judge_answer_quality(
                    query=query,
                    rag_context=rag.get("retrieved_context", ""),
                    rag_answer=rag.get("generated_answer", ""),
                    agentic_context=agentic.get("retrieved_context_str", ""),
                    agentic_answer=agentic.get("generated_answer", ""),
                )
                comparison_results[idx].update({
                    "answer_winner": result["winner"],
                    "answer_reasoning": result.get("reasoning", ""),
                })
                print(f"[Answer] Winner: {result['winner']} | {result.get('reasoning', '')}")
            except Exception as e:
                print(f"[Answer] ERROR: {e}")
                # FIX 5: Save error state
                comparison_results[idx].update({
                    "answer_winner": "ERROR",
                    "answer_reasoning": str(e),
                })
        else:
            print(f"[Answer] Already judged → {comparison_results[idx]['answer_winner']}. Skipping.")

        # Incremental save after every item
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=4)

        print()
        time.sleep(3)

    print("Comparison complete!\n")
    return comparison_results

if __name__ == "__main__":
    RESULTS_FILE = "comparison_results_multi.json"
    AGENTIC_SYSTEM_FILEPATH = "agentic_rag_results_multi.json"
    RAG_SYSTEM_FILEPATH = "rag_system_results_multi.json"

    run_comparison(RESULTS_FILE, AGENTIC_SYSTEM_FILEPATH, RAG_SYSTEM_FILEPATH)