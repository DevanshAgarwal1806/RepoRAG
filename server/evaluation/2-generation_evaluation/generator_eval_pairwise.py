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


def _call_judge(prompt: str) -> dict:
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

    Evaluate both answers on: faithfulness, correctness, completeness, and clarity.
    Declare the overall winner. If both answers are roughly equal in quality, declare a "Tie".

    Respond ONLY in this JSON format with no extra text:
    {{
    "winner": "<A | B | Tie>",
    "reasoning": "<One concise sentence explaining the key difference between the two answers>"
    }}"""
    return _call_judge(prompt)


def _answer_key(model_name: str) -> str:
    return f"{model_name}"

def run_comparison(model_1: str, model_2: str, results_filename: str) -> None:
    results_path = GENERATOR_EVALUATION_DIR / results_filename
    comparison_key = f"{model_1} vs {model_2}"

    with open(results_path, "r", encoding="utf-8") as f:
        gen_results = json.load(f)

    total = len(gen_results)

    print(f"\n{'='*60}")
    print(f"  {model_1}  vs  {model_2}")
    print(f"  File  : {results_filename}")
    print(f"  Items : {total}")
    print(f"{'='*60}\n")

    for idx, item in enumerate(gen_results):
        query = item.get("query", "")
        context = item.get("actual_context", "")
        answer_a = item.get(model_1, {}).get("answer", "")  # fix: {} not ""
        answer_b = item.get(model_2, {}).get("answer", "")  # fix: {} not ""

        print(f"[{idx+1}/{total}] Query")

        existing = item.get(comparison_key)
        if existing and not existing.startswith("ERROR"):
            print(f"Already judged: {existing}. Skipping.\n")
            continue

        if not answer_a or not answer_b:
            missing = model_1 if not answer_a else model_2
            print(f"SKIP — missing answer for '{missing}'")
            item[comparison_key] = f"ERROR — missing answer for {missing}"
        else:
            try:
                result = judge_answer_quality(
                    query=query,
                    context=context,
                    answer_a=answer_a,
                    answer_b=answer_b,
                )
                raw_winner = result["winner"]
                reasoning  = result.get("reasoning", "")

                winner_label = model_1 if raw_winner == "A" else (model_2 if raw_winner == "B" else "TIE")
                item[comparison_key] = {"winner": winner_label, "reasoning": reasoning}
                print(f"{comparison_key} = {item[comparison_key]}")

            except Exception as e:
                print(f"  ERROR: {e}")
                item[comparison_key] = f"ERROR — {e}"

            time.sleep(3)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(gen_results, f, indent=4)

        print()

if __name__ == "__main__":
    MODEL_A = "gemma3:4b"
    MODEL_B = "phi4-mini:3.8b"
    MODEL_C = "qwen2.5-coder:3b"

    for filename in ["rag_system_results_single.json", "rag_system_results_multi.json"]:
        run_comparison(MODEL_A, MODEL_B, filename)
        run_comparison(MODEL_A, MODEL_C, filename)
        run_comparison(MODEL_B, MODEL_C, filename)