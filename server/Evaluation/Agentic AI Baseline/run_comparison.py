"""
run_comparison.py
=================
Runs BOTH pipelines on the same ground-truth queries, then calls the
LLM judge on both result files so you get a clean apples-to-apples table.

Usage
-----
  # Step 1 – produce results for both pipelines
  python run_comparison.py run \
      --ground-truth  singlehop_ground_truth.json \
      --output        /path/to/repo_output \
      --your-system   your_system_results.json \
      --baseline      agentic_rag_results.json

  # Step 2 – score both with the LLM judge (relevance + faithfulness)
  python run_comparison.py judge \
      --your-system   your_system_results.json \
      --baseline      agentic_rag_results.json

  # Step 3 – print the comparison table
  python run_comparison.py report \
      --your-system   your_system_results.json \
      --baseline      agentic_rag_results.json
"""

import sys
import json
import argparse
import time
from pathlib import Path

# ── make sure both sibling folders are importable ────────────────────────────
COMP_DIR   = Path(__file__).resolve().parent
SERVER_DIR = COMP_DIR.parent.parent          # server/
sys.path.insert(0, str(SERVER_DIR))

from Evaluation.llm_as_judge import evaluate_pipeline_results, calculate_average_scores
from Evaluation.Agentic_AI_Baseline.agentic_rag import run_agentic_rag  # noqa (renamed below)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – your existing pipeline  (non-agentic, graph-augmented hybrid RAG)
# ══════════════════════════════════════════════════════════════════════════════

def run_your_system(query: str, output_dir: Path) -> dict:
    """
    Wraps your existing run_pipeline.py logic so it can be called per-query.
    Returns { query, retrieved_context, generated_answer }.
    """
    from hybrid_retrieval import run_hybrid_retrieval
    from retriever.generator import generate_rag_answer

    run_hybrid_retrieval(output_dir, query)          # writes final_llm_payload.md
    answer = generate_rag_answer(str(output_dir))    # reads it, calls Groq

    # Read the assembled context so the judge can score retrieval quality
    payload_path = output_dir / "final_llm_payload.md"
    context = payload_path.read_text(encoding="utf-8") if payload_path.exists() else ""

    return {
        "query":             query,
        "retrieved_context": context,
        "generated_answer":  answer or "",
    }


def batch_your_system(
    ground_truth_path: str,
    output_dir: str,
    results_path: str,
    query_key: str = "query",
):
    with open(ground_truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    try:
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
        done = {r["query"] for r in results}
        print(f"[Your System] Resuming: {len(results)} done.")
    except (FileNotFoundError, json.JSONDecodeError):
        results, done = [], set()

    out = Path(output_dir)
    for idx, item in enumerate(ground_truth):
        q = item.get(query_key, "")
        if not q or q in done:
            continue
        print(f"\n[Your System {idx+1}/{len(ground_truth)}] {q[:70]}…")
        try:
            r = run_your_system(q, out)
            results.append(r)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"[Your System] Done. {len(results)} results saved to {results_path}")


def batch_agentic(
    ground_truth_path: str,
    output_dir: str,
    results_path: str,
    query_key: str = "query",
):
    """Thin wrapper around agentic_rag.run_batch."""
    from Evaluation.Agentic_AI_Baseline import agentic_rag
    agentic_rag.run_batch(ground_truth_path, output_dir, results_path, query_key)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – judge both result files
# ══════════════════════════════════════════════════════════════════════════════

def judge_both(your_system_path: str, baseline_path: str):
    for label, path in [("Your System", your_system_path), ("Agentic RAG", baseline_path)]:
        print(f"\n{'='*60}")
        print(f"Judging: {label}  →  {path}")
        print('='*60)
        evaluate_pipeline_results(path, "relevance")
        time.sleep(5)    # give the API a brief rest between the two big batches
        evaluate_pipeline_results(path, "faithfulness")


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 – pretty comparison table
# ══════════════════════════════════════════════════════════════════════════════

def _load_scores(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rel  = [d.get("relevance_score",    0) for d in data]
    faith= [d.get("faithfulness_score", 0) for d in data]
    steps= [d.get("steps",              0) for d in data]  # 0 for non-agentic
    n    = len(data)
    return {
        "n":               n,
        "avg_relevance":   sum(rel)   / n if n else 0,
        "avg_faithfulness":sum(faith) / n if n else 0,
        "avg_steps":       sum(steps) / n if n else 0,
        "rel_2":           sum(1 for s in rel   if s == 2) / n if n else 0,
        "faith_2":         sum(1 for s in faith if s == 2) / n if n else 0,
    }


def print_report(your_system_path: str, baseline_path: str):
    ys = _load_scores(your_system_path)
    ag = _load_scores(baseline_path)

    def delta(a, b):
        d = a - b
        return f"+{d:.3f}" if d > 0 else f"{d:.3f}"

    print("\n" + "=" * 72)
    print(f"{'METRIC':<30} {'Your System':>14} {'Agentic RAG':>14} {'Δ (Yours−Base)':>14}")
    print("-" * 72)

    rows = [
        ("Queries evaluated",       f"{ys['n']}",     f"{ag['n']}",     ""),
        ("Avg Relevance   (0–2)",    f"{ys['avg_relevance']:.3f}",
                                     f"{ag['avg_relevance']:.3f}",
                                     delta(ys['avg_relevance'], ag['avg_relevance'])),
        ("Avg Faithfulness (0–2)",   f"{ys['avg_faithfulness']:.3f}",
                                     f"{ag['avg_faithfulness']:.3f}",
                                     delta(ys['avg_faithfulness'], ag['avg_faithfulness'])),
        ("Perfect Relevance   (=2)", f"{ys['rel_2']*100:.1f}%",
                                     f"{ag['rel_2']*100:.1f}%",
                                     delta(ys['rel_2'], ag['rel_2'])),
        ("Perfect Faithfulness (=2)",f"{ys['faith_2']*100:.1f}%",
                                     f"{ag['faith_2']*100:.1f}%",
                                     delta(ys['faith_2'], ag['faith_2'])),
        ("Avg Agent Steps",          "N/A (single-pass)",
                                     f"{ag['avg_steps']:.1f}",   ""),
    ]
    for name, v_ys, v_ag, d in rows:
        print(f"{name:<30} {v_ys:>14} {v_ag:>14} {d:>14}")

    print("=" * 72)
    print("\nInterpretation guide")
    print("  Δ > 0  →  Your system outperforms Agentic RAG on that metric")
    print("  Δ < 0  →  Agentic RAG outperforms your system on that metric")
    print("  Scores are judged by Gemini 2.5 Pro on a 0–2 scale")
    print("  (0=wrong, 1=partial, 2=perfect)\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare your system vs. Agentic RAG baseline")
    sub = parser.add_subparsers(dest="cmd")

    # ── run ──
    run_p = sub.add_parser("run", help="Run both pipelines on a ground-truth file")
    run_p.add_argument("--ground-truth", "-g", required=True)
    run_p.add_argument("--output",       "-o", required=True,
                       help="Indexed output dir (embeddings.json etc.)")
    run_p.add_argument("--your-system",  default="your_system_results.json")
    run_p.add_argument("--baseline",     default="agentic_rag_results.json")
    run_p.add_argument("--query-key",    default="query",
                       help="Key in ground-truth JSON that holds the query text")
    run_p.add_argument("--only",         choices=["yours", "agentic"],
                       help="Run only one pipeline (for debugging/resuming)")

    # ── judge ──
    judge_p = sub.add_parser("judge", help="Score both result files with the LLM judge")
    judge_p.add_argument("--your-system", default="your_system_results.json")
    judge_p.add_argument("--baseline",    default="agentic_rag_results.json")

    # ── report ──
    rep_p = sub.add_parser("report", help="Print the comparison table")
    rep_p.add_argument("--your-system", default="your_system_results.json")
    rep_p.add_argument("--baseline",    default="agentic_rag_results.json")

    args = parser.parse_args()

    if args.cmd == "run":
        if args.only != "agentic":
            print("\n── Running your system ──────────────────────────")
            batch_your_system(args.ground_truth, args.output, args.your_system, args.query_key)
        if args.only != "yours":
            print("\n── Running Agentic RAG baseline ─────────────────")
            from Evaluation.Agentic_AI_Baseline import agentic_rag
            agentic_rag.run_batch(args.ground_truth, args.output, args.baseline, args.query_key)

    elif args.cmd == "judge":
        judge_both(args.your_system, args.baseline)

    elif args.cmd == "report":
        print_report(args.your_system, args.baseline)

    else:
        parser.print_help()
