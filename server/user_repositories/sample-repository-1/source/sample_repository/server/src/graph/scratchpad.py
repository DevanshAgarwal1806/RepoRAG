# src/graph/scratchpad.py

from collections import Counter


def parse_scratchpad(scratchpad: list[str]) -> tuple[set[str], set[str]]:
    """
    Parses the reflexion scratchpad and returns (completed_ids, failed_ids).
    Expects entries in the format: STATUS|task_id|content|Reason: ...
    """
    completed = set()
    failed    = set()

    for entry in scratchpad:
        parts = entry.split("|")
        if len(parts) < 2:
            continue
        status  = parts[0]
        task_id = parts[1]
        if status == "SUCCESS":
            completed.add(task_id)
        elif status == "FAILURE":
            failed.add(task_id)

    return completed, failed


def get_exhausted_task_ids(scratchpad: list[str], max_retries: int) -> set[str]:
    """
    Returns tasks that have failed enough times to be treated as terminally failed.
    """
    failure_counts: Counter[str] = Counter()

    for entry in scratchpad:
        if not entry.startswith("FAILURE|"):
            continue

        parts = entry.split("|")
        if len(parts) < 2:
            continue

        failure_counts[parts[1]] += 1

    return {
        task_id
        for task_id, count in failure_counts.items()
        if count >= max_retries
    }


def get_attempted_tools(scratchpad: list[str], task_id: str) -> list[str]:
    """
    Returns the ordered list of tool names already attempted for a task.
    Expects entries in the format: ATTEMPT|task_id|tool_name|tool_input
    """
    attempted_tools: list[str] = []

    for entry in scratchpad:
        if not entry.startswith(f"ATTEMPT|{task_id}|"):
            continue

        parts = entry.split("|", 3)
        if len(parts) < 3:
            continue

        tool_name = parts[2]
        if tool_name not in attempted_tools:
            attempted_tools.append(tool_name)

    return attempted_tools
