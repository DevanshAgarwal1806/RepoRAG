import sys
import json
from pathlib import Path
import time
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

GENERATOR_EVALUATION_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = GENERATOR_EVALUATION_DIR.parent
SERVER_DIR = EVALUATION_DIR.parent

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

JUDGE_MODEL = "gemma-4-31b-it"
ANSWER_CRITERIA = ["faithfulness", "correctness", "completeness", "clarity"]


def _call_judge(prompt: str) -> dict:
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

    winner = output.get("winner", "").strip().upper()
    if winner not in ("A", "B", "TIE"):
        raise ValueError(f"Unexpected winner value: '{winner}'")
    output["winner"] = winner

    scores = output.get("scores", {})
    for system in ("A", "B"):
        if system not in scores:
            raise ValueError(f"Missing scores for system '{system}'")
        for criterion in ANSWER_CRITERIA:
            if criterion not in scores[system]:
                raise ValueError(f"Missing score '{criterion}' for system '{system}'")
            score = scores[system][criterion]
            if not isinstance(score, (int, float)) or not (0 <= score <= 2):
                raise ValueError(f"Score '{criterion}' for '{system}' must be 0-2, got: {score}")

    return output


def judge_answer_quality(query: str, context: str, answer_a: str, answer_b: str) -> dict:
    prompt = f"""You are an expert code reviewer. Your task is to judge which AI-generated answer better addresses the user query, given the same shared code context.
    Judge each answer ONLY against the provided context — penalise any answer that hallucinates logic not present in the retrieved code.

    User Query:
    {query}

    Shared Retrieved Context:
    {context}

    === Answer A ===
    {answer_a}

    === Answer B ===
    {answer_b}

    Score EACH answer independently on all four criteria using this 0-2 scale:
    0 = Poor    — Does not meet the criterion at all
    1 = Partial — Partially meets the criterion
    2 = Good    — Fully meets the criterion

    Criteria definitions (in order of priority):
    faithfulness — Is the answer fully grounded in the retrieved context with no hallucinations?
    correctness  — Is the answer technically accurate?
    completeness — Does it fully address all parts of the query?
    clarity      — Is the explanation clear, structured, and easy to follow?

    After scoring both answers, declare the overall winner based on total scores.
    If total scores differ by at most 1 (out of 8), declare a "Tie".

    Respond ONLY in this JSON format with no extra text:
    {{
    "scores": {{
        "A": {{
        "faithfulness": <0|1|2>,
        "correctness":  <0|1|2>,
        "completeness": <0|1|2>,
        "clarity":      <0|1|2>
        }},
        "B": {{
        "faithfulness": <0|1|2>,
        "correctness":  <0|1|2>,
        "completeness": <0|1|2>,
        "clarity":      <0|1|2>
        }}
    }},
    "winner": "<A | B | Tie>",
    "reasoning": "<One concise sentence explaining the key difference between the two answers>"
    }}"""
    return _call_judge(prompt)


def _answer_key(model_name: str) -> str:
    return f"generated_answer_{model_name}"

def _output_filename(model_1: str, model_2: str, results_filename: str) -> str:
    suffix = results_filename.replace("rag_system_results_", "").replace(".json", "")
    m1 = model_1.replace(":", "-").replace("/", "-")
    m2 = model_2.replace(":", "-").replace("/", "-")
    return f"comparison_{m1}_vs_{m2}_{suffix}.json"

def run_comparison(model_1: str, model_2: str, results_filename: str) -> list:
    results_path = GENERATOR_EVALUATION_DIR / results_filename
    output_filename = _output_filename(model_1, model_2, results_filename)
    output_path = GENERATOR_EVALUATION_DIR / output_filename

    with open(results_path, "r", encoding="utf-8") as f:
        rag_results = json.load(f)

    total = len(rag_results)

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            comparison_results = json.load(f)
    else:
        comparison_results = []

    while len(comparison_results) < total:
        comparison_results.append({})

    key_a = _answer_key(model_1)
    key_b = _answer_key(model_2)

    print(f"\n{'='*60}")
    print(f"  {model_1}  vs  {model_2}")
    print(f"  File   : {results_filename}")
    print(f"  Output : {output_filename}")
    print(f"  Items  : {total}")
    print(f"{'='*60}\n")

    for idx, item in enumerate(rag_results):
        query    = item.get("query", "")
        context  = item.get("retrieved_context", "")
        answer_a = item.get(key_a, "")
        answer_b = item.get(key_b, "")

        print(f"[{idx+1}/{total}] Query: {query[:70]}...")

        comparison_results[idx]["query"] = query

        if comparison_results[idx].get("winner") in (None, "ERROR"):
            if not answer_a or not answer_b:
                missing = model_1 if not answer_a else model_2
                print(f"  SKIP — missing answer for '{missing}'")
                comparison_results[idx].update({
                    "winner":    "ERROR",
                    "reasoning": f"Missing answer for {missing}",
                    "scores":    {},
                })
            else:
                try:
                    result = judge_answer_quality(
                        query=query,
                        context=context,
                        answer_a=answer_a,
                        answer_b=answer_b,
                    )
                    comparison_results[idx].update({
                        "winner":    result["winner"],
                        "reasoning": result.get("reasoning", ""),
                        "scores":    result.get("scores", {}),
                    })
                    print(f"  Winner: {result['winner']} | {result.get('reasoning', '')}")
                    print(f"  Scores {model_1}: {result['scores']['A']}")
                    print(f"  Scores {model_2}: {result['scores']['B']}")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    comparison_results[idx].update({
                        "winner": "ERROR",
                        "reasoning": str(e),
                        "scores": {},
                    })

            time.sleep(3)
        else:
            print(f"  Already judged {comparison_results[idx]['winner']}. Skipping.")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=4)

        print()

    print(f"Done. Results saved to: {output_filename}\n")
    return comparison_results

def aggregate_scores(comparison_results_filepath: str) -> None:
    comparison_results_filepath = GENERATOR_EVALUATION_DIR / comparison_results_filepath
    with open(comparison_results_filepath, "r", encoding="utf-8") as f:
        comparison_results = json.load(f)

    num_queries = len(comparison_results)

    answer_scores_dict = {
        "A": {criterion: 0 for criterion in ANSWER_CRITERIA},
        "B": {criterion: 0 for criterion in ANSWER_CRITERIA},
    }

    for item in comparison_results:
        scores = item.get("scores", {})
        for system in ("A", "B"):
            for criterion in ANSWER_CRITERIA:
                answer_scores_dict[system][criterion] += scores.get(system, {}).get(criterion, 0)

    for system in ("A", "B"):
        for criterion in ANSWER_CRITERIA:
            answer_scores_dict[system][criterion] /= num_queries

    comparison_results.append({"aggregate_answer_scores": answer_scores_dict})

    with open(comparison_results_filepath, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=4)

if __name__ == "__main__":
    MODEL_A = "llama-3.3-70b-versatile"
    MODEL_B = "qwen2.5-coder:7b"
    MODEL_C = "gemma3:4b"

    for filename in ["rag_system_results_single.json", "rag_system_results_multi.json"]:
        run_comparison(MODEL_A, MODEL_B, filename)
        run_comparison(MODEL_A, MODEL_C, filename)
        run_comparison(MODEL_B, MODEL_C, filename)
        
        print(f"\nAggregating scores for {filename}...")
        aggregate_scores(_output_filename(MODEL_A, MODEL_B, filename))
        aggregate_scores(_output_filename(MODEL_A, MODEL_C, filename))
        aggregate_scores(_output_filename(MODEL_B, MODEL_C, filename))