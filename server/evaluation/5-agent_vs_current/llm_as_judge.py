import json
import time
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parent
SERVER_DIR = EVALUATION_DIR.parent
GROUND_TRUTH_DIR = EVALUATION_DIR / "0-ground_truth_construction"
DEFAULT_OUTPUT_DIR = SERVER_DIR / "sample_repository_output"
COMPLETE_SYSTEM_EVALUATION_DIR = EVALUATION_DIR / "3-complete_system_evaluation"
AGENTIC_SYSTEM_OUTPUT_DIR = EVALUATION_DIR / "4-agentic_rag_baseline"

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

JUDGE_MODEL = "gemma-4-31b-it"

# Scoring criteria and their weights for the final score
ANSWER_CRITERIA = ["faithfulness", "correctness", "completeness", "clarity"]
CONTEXT_CRITERIA = ["relevance", "completeness", "precision"]

def _call_judge(prompt: str, expected_criteria: list[str]) -> dict:
    """
    Calls Gemma 4 31B, parses JSON, and validates all expected score fields exist.
    Raises on failure.
    """
    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=1024,
        ),
    )
    output = json.loads(response.text)

    # Validate winner
    winner = output.get("winner", "").strip().upper()
    if winner not in ("A", "B", "TIE"):
        raise ValueError(f"Unexpected winner value: '{winner}'")
    output["winner"] = winner

    # Validate scores block exists for both systems
    scores = output.get("scores", {})
    for system in ("A", "B"):
        if system not in scores:
            raise ValueError(f"Missing scores for system '{system}'")
        for criterion in expected_criteria:
            if criterion not in scores[system]:
                raise ValueError(f"Missing score '{criterion}' for system '{system}'")
            score = scores[system][criterion]
            if not isinstance(score, (int, float)) or not (0 <= score <= 5):
                raise ValueError(f"Score '{criterion}' for '{system}' must be 0-5, got: {score}")

    return output


def judge_context_relevance(query: str, rag_context: str, agentic_context: str) -> dict:
    """
    Head-to-head: which system retrieved more relevant code snippets for the query?
    A = RAG system, B = Agentic system
    Scores each system on relevance, completeness, precision (0-5 each).
    """
    prompt = f"""You are an expert code search evaluator. Your task is to judge which retrieved code context is MORE RELEVANT and USEFUL for answering the given user query.

    User Query:
    {query}

    --- Context A ---
    {rag_context}

    --- Context B ---
    {agentic_context}

    Score EACH system independently on all three criteria using this 0-5 scale:
    0 = Completely absent — criterion not met at all
    1 = Minimal          — very weak signal, mostly irrelevant
    2 = Poor             — some relevance but largely inadequate
    3 = Partial          — meets the criterion to a moderate degree
    4 = Good             — mostly meets the criterion with minor gaps
    5 = Excellent        — fully and precisely meets the criterion

    Criteria definitions:
    relevance    — Does the context directly contain code related to the query?
    completeness — Does it capture the full logic needed (not just fragments)?
    precision    — Is the retrieved code focused with minimal noise/unrelated code?

    After scoring both systems, declare the overall winner based on total scores.
    If total scores are equal or differ by at most 2 (out of 15), declare a "Tie".

    Respond ONLY in this JSON format with no extra text:
    {{
    "scores": {{
        "A": {{
        "relevance": <0|1|2|3|4|5>,
        "completeness": <0|1|2|3|4|5>,
        "precision": <0|1|2|3|4|5>
        }},
        "B": {{
        "relevance": <0|1|2|3|4|5>,
        "completeness": <0|1|2|3|4|5>,
        "precision": <0|1|2|3|4|5>
        }}
    }},
    "winner": "<A | B | Tie>",
    "reasoning": "<One concise sentence explaining your decision>"
    }}"""
    return _call_judge(prompt, expected_criteria=CONTEXT_CRITERIA)


def judge_answer_quality(
    query: str,
    rag_context: str, rag_answer: str,
    agentic_context: str, agentic_answer: str
) -> dict:
    """
    Head-to-head: which system generated a better answer, grounded in its own retrieved context?
    A = RAG system, B = Agentic system
    Scores each system on faithfulness, correctness, completeness, clarity (0-5 each).
    """
    prompt = f"""You are an expert code reviewer. Your task is to judge which AI system produced a BETTER answer to the user query.
    Each system has its own retrieved code context and generated answer. Judge each answer ONLY against its own context — penalise any system that hallucinates logic not present in its retrieved code.

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

    Score EACH system independently on all four criteria using this 0-5 scale:
    0 = Completely absent — criterion not met at all
    1 = Minimal          — very weak signal, mostly irrelevant
    2 = Poor             — some relevance but largely inadequate
    3 = Partial          — meets the criterion to a moderate degree
    4 = Good             — mostly meets the criterion with minor gaps
    5 = Excellent        — fully and precisely meets the criterion

    Criteria definitions (evaluated in order of priority):
    faithfulness — Is the answer fully grounded in its own retrieved context with no hallucinations?
    correctness  — Is the answer technically accurate?
    completeness — Does it fully address all parts of the query?
    clarity      — Is the explanation clear, structured, and easy to follow?

    After scoring both systems, declare the overall winner based on total scores.
    If total scores differ by at most 3 (out of 20), declare a "Tie".

    Respond ONLY in this JSON format with no extra text:
    {{
    "scores": {{
        "A": {{
        "faithfulness": <0|1|2|3|4|5>,
        "correctness": <0|1|2|3|4|5>,
        "completeness": <0|1|2|3|4|5>,
        "clarity": <0|1|2|3|4|5>
        }},
        "B": {{
        "faithfulness": <0|1|2|3|4|5>,
        "correctness": <0|1|2|3|4|5>,
        "completeness": <0|1|2|3|4|5>,
        "clarity": <0|1|2|3|4|5>
        }}
    }},
    "winner": "<A | B | Tie>",
    "reasoning": "<One concise sentence explaining your decision>"
    }}"""
    return _call_judge(prompt, expected_criteria=ANSWER_CRITERIA)


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
                    "context_scores": result.get("scores", {}),
                })
                print(f"[Context] Winner: {result['winner']} | {result.get('reasoning', '')}")
                print(f"          Scores A: {result['scores']['A']}")
                print(f"          Scores B: {result['scores']['B']}")
            except Exception as e:
                print(f"[Context] ERROR: {e}")
                comparison_results[idx].update({
                    "context_winner": "ERROR",
                    "context_reasoning": str(e),
                    "context_scores": {},
                })

            time.sleep(3)
        else:
            print(f"[Context] Already judged {comparison_results[idx]['context_winner']}. Skipping.")

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
                    "answer_scores": result.get("scores", {}),
                })
                print(f"[Answer] Winner: {result['winner']} | {result.get('reasoning', '')}")
                print(f"         Scores A: {result['scores']['A']}")
                print(f"         Scores B: {result['scores']['B']}")
            except Exception as e:
                print(f"[Answer] ERROR: {e}")
                comparison_results[idx].update({
                    "answer_winner": "ERROR",
                    "answer_reasoning": str(e),
                    "answer_scores": {},
                })
        else:
            print(f"[Answer] Already judged {comparison_results[idx]['answer_winner']}. Skipping.")

        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=4)

        print()
        time.sleep(3)

    print("Comparison complete!\n")
    return comparison_results

def aggregate_scores(comparison_results_filepath: str) -> None:
    comparison_results_filepath = EVALUATION_DIR / comparison_results_filepath
    with open(comparison_results_filepath, "r", encoding="utf-8") as f:
        comparison_results = json.load(f)
        
    num_queries = len(comparison_results)
        
    context_scores_dict = {
        "A": {criterion: 0 for criterion in CONTEXT_CRITERIA},
        "B": {criterion: 0 for criterion in CONTEXT_CRITERIA},
    }

    for item in comparison_results:
        scores = item.get("context_scores", {})
        for system in ("A", "B"):
            for criterion in CONTEXT_CRITERIA:
                context_scores_dict[system][criterion] += scores.get(system, {}).get(criterion, 0)
        
    for system in ("A", "B"):
        for criterion in CONTEXT_CRITERIA:
            context_scores_dict[system][criterion] /= num_queries
    
    answer_scores_dict = {
        "A": {criterion: 0 for criterion in ANSWER_CRITERIA},
        "B": {criterion: 0 for criterion in ANSWER_CRITERIA},
    }

    for item in comparison_results:
        scores = item.get("answer_scores", {})
        for system in ("A", "B"):
            for criterion in ANSWER_CRITERIA:
                answer_scores_dict[system][criterion] += scores.get(system, {}).get(criterion, 0)
        
    for system in ("A", "B"):
        for criterion in ANSWER_CRITERIA:
            answer_scores_dict[system][criterion] /= num_queries

    comparison_results.append({"aggregate_context_scores": context_scores_dict})
    comparison_results.append({"aggregate_answer_scores": answer_scores_dict})

    with open(comparison_results_filepath, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=4)

if __name__ == "__main__":
    RESULTS_FILE = "comparison_results_multi.json"
    AGENTIC_SYSTEM_FILE = "agentic_rag_results_multi.json"
    RAG_SYSTEM_FILE = "rag_evaluation_results_multi.json"

    run_comparison(RESULTS_FILE, AGENTIC_SYSTEM_FILE, RAG_SYSTEM_FILE)
    print("Aggregating scores...")
    aggregate_scores(RESULTS_FILE)

    RESULTS_FILE = "comparison_results_single.json"
    AGENTIC_SYSTEM_FILE = "agentic_rag_results_single.json"
    RAG_SYSTEM_FILE = "rag_evaluation_results_single.json"

    run_comparison(RESULTS_FILE, AGENTIC_SYSTEM_FILE, RAG_SYSTEM_FILE)
    print("Aggregating scores...")
    aggregate_scores(RESULTS_FILE)