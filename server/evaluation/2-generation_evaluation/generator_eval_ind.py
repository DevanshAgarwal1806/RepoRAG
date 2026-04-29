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
SCORE_METRICS = ["faithfulness", "completeness", "correctness", "clarity"]


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

    scores = output.get("scores", {})
    for metric in SCORE_METRICS:
        if metric not in scores:
            raise ValueError(f"Missing metric '{metric}' in scores")
        val = scores[metric]
        if not isinstance(val, (int, float)) or not (1 <= val <= 5):
            raise ValueError(f"Score for '{metric}' must be 1-5, got: {val}")

    return output


def score_answer(query: str, context: str, answer: str) -> dict:
    prompt = f"""You are an expert code reviewer. Your task is to evaluate an AI-generated answer to a query about a codebase.
    Judge the answer ONLY against the provided context — penalise any answer that hallucinates logic not present in the retrieved code.

    User Query:
    {query}

    Retrieved Code Context:
    {context}

    === Answer to Evaluate ===
    {answer}

    Score the answer on each of the following metrics using a 1-5 scale:
    1 = Very poor   2 = Poor   3 = Average   4 = Good   5 = Excellent

    Metric definitions:
    faithfulness  — Is the answer fully grounded in the retrieved context with no hallucinations?
    completeness  — Does it fully address all parts of the query?
    correctness   — Is the answer technically accurate?
    clarity       — Is the explanation clear and easy to follow?

    Respond ONLY in this JSON format with no extra text:
    {{
        "scores": {{
            "faithfulness":  <1|2|3|4|5>,
            "completeness":  <1|2|3|4|5>,
            "correctness":   <1|2|3|4|5>,
            "clarity":       <1|2|3|4|5>
        }},
        "reasoning": "<One concise sentence summarising the answer's overall quality>"
    }}"""
    return _call_judge(prompt)

def aggregate_scores(models: list[str], gen_results: list[dict]) -> dict:
    aggregated = {}

    for model in models:
        metric_totals = {metric: 0.0 for metric in SCORE_METRICS}
        count = 0

        for item in gen_results:
            model_data = item.get(model)
            if not isinstance(model_data, dict):
                continue
            scores = model_data.get("scores")
            if not isinstance(scores, dict):
                continue  # skip errored or missing scores

            if all(metric in scores for metric in SCORE_METRICS):
                for metric in SCORE_METRICS:
                    metric_totals[metric] += scores[metric]
                count += 1

        if count == 0:
            aggregated[model] = {"error": "No valid scores found"}
            continue

        avg_scores = {metric: round(metric_totals[metric] / count, 3) for metric in SCORE_METRICS}
        aggregated[model] = avg_scores

    return aggregated

def run_scoring(models: list[str], results_filename: str) -> None:
    results_path = GENERATOR_EVALUATION_DIR / results_filename

    with open(results_path, "r", encoding="utf-8") as f:
        gen_results = json.load(f)

    total = len(gen_results)

    for idx, item in enumerate(gen_results):
        query   = item.get("query", "")
        context = item.get("actual_context", "")

        print(f"[{idx+1}/{total}] Query")

        for model in models:
            model_data = item.get(model)
            if not isinstance(model_data, dict):
                print(f"  SKIP {model} — no data found")
                continue

            if model_data.get("scores"):
                if isinstance(model_data.get("scores"), str) and model_data.get("scores").startswith("ERROR"):
                     print(f"  {model}: previous error — {model_data['scores']}")
                else:
                    print(f"  {model}: already scored, skipping")
                    continue

            answer = model_data.get("answer", "")
            if not answer:
                print(f"  SKIP {model} — empty answer")
                continue

            try:
                result = score_answer(query=query, context=context, answer=answer)
                model_data["scores"] = result["scores"]
                print(f"  {model}: {result['scores']}  | {result.get('reasoning', '')}")
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(gen_results, f, indent=4)
            except Exception as e:
                print(f"  ERROR scoring {model}: {e}")
                model_data["scores"] = f"ERROR — {e}"

            time.sleep(3)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(gen_results, f, indent=4)

        print()

    print(f"Done. Results saved to: {results_filename}\n")
    
    print("Aggregating scores...")
    aggregated = aggregate_scores(models, gen_results)
    
    for model, scores in aggregated.items():
        print(f"{model}: {scores}")


if __name__ == "__main__":
    MODELS = ["gemma3:4b", "phi4-mini:3.8b", "qwen2.5-coder:3b", "llama-3.3-70b-versatile"]

    for filename in ["gen_results_single.json", "gen_results_multi.json"]:
        run_scoring(MODELS, filename)