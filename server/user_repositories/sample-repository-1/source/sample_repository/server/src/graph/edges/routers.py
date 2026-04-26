from src.graph.state import SynapseState
from src.graph.scratchpad import parse_scratchpad
from src.config.settings import (
    DAG_MAX_RETRIES,
    DAG_PASSING_SCORE,
    TASK_MAX_RETRIES,
)


def route_evaluation(state: SynapseState) -> str:
    score      = state.get("judge_score", 0)
    loop_count = state.get("loop_count", 0)

    if score >= DAG_PASSING_SCORE:
        print(f"--- DAG APPROVED (Score: {score}/15) -> PROCEEDING TO EXECUTION ---")
        return "execute"
    elif loop_count >= DAG_MAX_RETRIES:
        print(f"--- MAX RETRIES REACHED ({DAG_MAX_RETRIES}). FORCING EXECUTION ---")
        return "execute"
    else:
        print(f"--- DAG REJECTED (Score: {score}/15, attempt {loop_count}/{DAG_MAX_RETRIES}). RETRYING ---")
        return "generate"


def route_execution(state: SynapseState) -> str:
    task_id    = state.get("current_step_id", "")
    scratchpad = state.get("reflexion_scratchpad", [])
    dag        = state.get("current_dag", {})

    # Hard stop — prevent infinite loops
    max_entries = len(dag.get("tasks", [])) * 6
    if len(scratchpad) > max_entries:
        print(f"[FATAL] Scratchpad has {len(scratchpad)} entries — "
              f"infinite loop detected. Forcing exit.")
        return "done"

    if task_id == "DONE":
        return "done"

    completed_ids, failed_ids = parse_scratchpad(scratchpad)

    # Count retries for the current task only
    retry_count = sum(
        1 for e in scratchpad
        if e.startswith(f"FAILURE|{task_id}|")    # ← pipe throughout
    )

    # Find the most recent verdict for this task
    last_verdict = ""
    for entry in reversed(scratchpad):
        if entry.startswith(f"SUCCESS|{task_id}|") or \
           entry.startswith(f"FAILURE|{task_id}|"):    # ← pipe throughout
            last_verdict = entry
            break

    if not last_verdict:
        print(f"[WARNING] No verdict found for task {task_id}. Forcing done.")
        return "done"

    if last_verdict.startswith(f"SUCCESS|{task_id}|"):    # ← pipe throughout
        total_tasks = len(dag.get("tasks", []))
        if len(completed_ids) >= total_tasks:
            print("--- All tasks complete → DONE ---")
            return "done"
        print(f"--- Task {task_id} succeeded → NEXT TASK ---")
        return "next_task"

    # It was a failure
    if retry_count < TASK_MAX_RETRIES:
        print(f"--- Task {task_id} failed "
              f"(attempt {retry_count + 1}/{TASK_MAX_RETRIES}) → RETRYING ---")
        return "retry"

    print(f"--- Task {task_id} exhausted retries → FAIL ---")
    return "fail"