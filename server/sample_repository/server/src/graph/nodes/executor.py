from src.graph.state import SynapseState
from src.tools.registry import get_all_tools
from src.tools.tool_selector import select_tool
from src.graph.scratchpad import (
    get_attempted_tools,
    get_exhausted_task_ids,
    parse_scratchpad,
)
from src.config.settings import TASK_MAX_RETRIES

RAW_RESULT_LIMIT = 6000


def _get_pending_task(dag: dict, scratchpad: list) -> dict | None:
    completed_ids, _ = parse_scratchpad(scratchpad)
    exhausted_ids = get_exhausted_task_ids(scratchpad, TASK_MAX_RETRIES)

    for task in dag.get("tasks", []):
        task_id = task["id"]
        if task_id in completed_ids or task_id in exhausted_ids:
            continue
        if all(dep in completed_ids for dep in task.get("dependencies", [])):
            return task

    return None


def execute_task(state: SynapseState) -> dict:
    print("--- EXECUTING TASK ---")

    dag        = state.get("current_dag", {})
    scratchpad = state.get("reflexion_scratchpad", [])

    task = _get_pending_task(dag, scratchpad)

    if task is None:
        print("--- ALL TASKS COMPLETE ---")
        return {
            "current_step_id": "DONE",
            "final_output":    _compile_final_output(scratchpad),
        }

    task_id = task["id"]
    print(f"--- Task: {task_id} | {task.get('description', '')} ---")

    tool_map = {t.name: t for t in get_all_tools()}
    attempted_tools = get_attempted_tools(scratchpad, task_id)

    if attempted_tools:
        print(f"--- Excluding previously attempted tools: {', '.join(attempted_tools)} ---")

    # select_tool returns (tool, tool_input, warnings)
    tool, tool_input, warnings = select_tool(
        task,
        tool_map,
        excluded_tool_names=set(attempted_tools),
    )

    if tool is None:
        return {
            "current_step_id":      task_id,
            "execution_warnings":   warnings,
            "reflexion_scratchpad": [
                f"FAILURE|{task_id}|No suitable tool found after generalization.|Reason: tool_selector exhausted"
            ],                                                  # ← pipe after task_id
        }

    print(f"--- Running: {tool.name}({tool_input!r}) ---")

    try:
        raw_result = str(tool.invoke(tool_input))
    except Exception as e:
        raw_result = f"TOOL_ERROR: {str(e)}"

    return {
        "current_step_id":      task_id,
        "execution_warnings":   warnings,
        "reflexion_scratchpad": [
            f"ATTEMPT|{task_id}|{tool.name}|{tool_input}",
            f"RAW_RESULT|{task_id}|{raw_result[:RAW_RESULT_LIMIT]}",
        ],
    }


def _compile_final_output(scratchpad: list) -> str:
    lines = [e for e in scratchpad if e.startswith("SUCCESS|")]
    return "\n\n".join(lines) if lines else "Execution complete. No successful outputs recorded."
