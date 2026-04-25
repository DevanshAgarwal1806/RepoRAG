from dotenv import load_dotenv
load_dotenv()
import sys
from src.graph.workflow import build_synapse_graph

# ── Fully initialised state ───────────────────────────────────────────────────
# Every key in SynapseState must be present at invoke time.
# Missing keys cause a KeyError on the first node that reads them.

def build_initial_state(prompt: str) -> dict:
    return {
        "user_prompt":          prompt,
        "current_dag":          {},
        "judge_score":          0,
        "feedback":             "",
        "loop_count":           0,
        "current_step_id":      "",
        "reflexion_scratchpad": [],
        "final_output":         None,
        "useful_output":        True,
        "execution_warnings":   [],
    }


# ── Warning display ─────────────────────────────────────────────────────────

def print_warnings(warnings: list[dict]) -> None:
    if not warnings:
        return

    sep = "═" * 64
    print(f"\n{sep}")
    print("  EXECUTION WARNINGS")
    print(sep)

    for w in warnings:
        print(f"\n  Task ID : {w['task_id']}")
        print(f"  For task with description : \"{w['original']}\"")

        if w["tool_chosen"]:
            print(f"  No appropriate tool was found and hence the prompt")
            print(f"  was simplified to this    : \"{w['simplified']}\"")
            print(f"  Tool used                 : {w['tool_chosen']}")
            print(f"  Simplification steps      : {w['steps_taken']}")
        else:
            print(f"  No appropriate tool was found after "
                  f"{w['steps_taken']} simplification attempt(s).")
            print(f"  Final simplified prompt   : \"{w['simplified']}\"")
            print(f"  This task was SKIPPED.")

    print(f"\n{sep}\n")


# ── Final output display ──────────────────────────────────────────────────────

# def print_final_output(result: dict) -> None:
#     sep = "═" * 64
#     print(f"\n{sep}")
#     print("  FINAL OUTPUT")
#     print(sep)

#     output = result.get("final_output")
#     if output:
#         print(f"\n{output}\n")
#     else:
#         # final_output is only set when executor hits DONE cleanly.
#         # If the graph ended via the 'fail' route it may be None —
#         # fall back to printing whatever succeeded in the scratchpad.
#         scratchpad = result.get("reflexion_scratchpad", [])
#         successes = [e for e in scratchpad if e.startswith("SUCCESS|")]

#         if successes:
#             print("\nPartial results from completed tasks:\n")
#             for entry in successes:
#                 parts = entry.split("|")
#                 task_id  = parts[1] if len(parts) > 1 else "unknown"
#                 content  = parts[2] if len(parts) > 2 else ""
#                 print(f"  [{task_id}] {content}\n")
#         else:
#             print("\n  No output produced. Check warnings above.\n")

#     print(sep)

def print_final_output(result: dict) -> None:
    sep = "═" * 64
    print(f"\n{sep}")
    print("  FINAL OUTPUT")
    print(sep)
    print(f"\n{result.get('final_output', 'No output produced.')}\n")
    print(sep)
# ── Debug helpers (optional, remove in production) ───────────────────────────

def print_dag(dag: dict) -> None:
    import json
    print("\n── Approved DAG ──────────────────────────────────────────")
    print(json.dumps(dag, indent=2))
    print("──────────────────────────────────────────────────────────\n")


def print_scratchpad(scratchpad: list[str]) -> None:
    print("\n── Reflexion Scratchpad ──────────────────────────────────")
    for i, entry in enumerate(scratchpad, 1):
        # Truncate long raw results so the terminal stays readable
        display = entry if len(entry) < 200 else entry[:200] + "...[truncated]"
        print(f"  {i:02d}. {display}")
    print("──────────────────────────────────────────────────────────\n")


# ── Runner ────────────────────────────────────────────────────────────────────

def run(prompt: str, debug: bool = False) -> dict:
    print(f"\nSynapseAI — starting run")
    print(f"Prompt: {prompt}\n")
    for i in range (2):
        graph = build_synapse_graph()
        initial_state = build_initial_state(prompt)

        try:
            result = graph.invoke(initial_state)
        except Exception as e:
            print(f"\n[FATAL] Graph execution failed: {e}")
            raise
        if result['useful_output'] == False:
            continue
        if debug:
            print_dag(result.get("current_dag", {}))
            print_scratchpad(result.get("reflexion_scratchpad", []))

        print_warnings(result.get("execution_warnings", []))
        print_final_output(result)
        
        return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Usage:
    #   python main.py "your prompt here"
    #   python main.py "your prompt here" --debug

    if len(sys.argv) < 2:
        print("Usage: python main.py \"your prompt\" [--debug]")
        sys.exit(1)

    user_prompt = sys.argv[1]
    debug_mode  = "--debug" in sys.argv

    run(user_prompt, debug=debug_mode)