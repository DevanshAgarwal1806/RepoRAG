import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.graph.state import SynapseState
from src.config.prompts import REFLECTOR_SYSTEM_PROMPT
from src.config.keypool import pool         


def reflect_on_execution(state: SynapseState) -> dict:
    print("--- REFLECTING ON RESULT ---")

    scratchpad = state.get("reflexion_scratchpad", [])
    task_id    = state.get("current_step_id", "unknown")
    dag        = state.get("current_dag", {})

    # Find the most recent RAW_RESULT for this task
    raw_entry = ""
    for entry in reversed(scratchpad):
        if entry.startswith(f"RAW_RESULT|{task_id}|"):          # ← pipe not colon
            raw_entry = entry[len(f"RAW_RESULT|{task_id}|"):]   # ← colon slice, no stray |
            break

    if not raw_entry:
        return {
            "reflexion_scratchpad": [
                f"FAILURE|{task_id}|No raw result found.|Reason: missing RAW_RESULT entry"
            ]
        }

    task_obj         = next(
        (t for t in dag.get("tasks", []) if t["id"] == task_id), {}
    )
    task_description = task_obj.get("description", task_obj.get("name", task_id))

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        api_key=pool.next()
    )

    messages = [
        SystemMessage(content=REFLECTOR_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Task ID: {task_id}\n"
            f"Task objective: {task_description}\n\n"
            f"Raw tool output:\n{raw_entry}\n\n"
            "Respond with a JSON object containing "
            "{\"status\": \"SUCCESS\" or \"FAILURE\", "
            "\"headline\": \"the single most important result\", "
            "\"detailed_summary\": \"a rich factual summary with concrete details\", "
            "\"reason\": \"a short explanation\"}. "
            "Prefer SUCCESS if the output is reasonably relevant or partially correct. "
            "Return FAILURE only if the output clearly does not address the task or is unusable. "
            "Ensure the response is valid JSON."
        ))
    ]

    response     = llm.invoke(messages)
    raw_response = response.content.strip()

    start = raw_response.find('{')
    end   = raw_response.rfind('}')
    verdict = {
        "status": "FAILURE",
        "headline": "",
        "detailed_summary": "",
        "reason": "Parse error",
    }

    if start != -1 and end != -1:
        try:
            verdict = json.loads(raw_response[start:end + 1])
        except Exception as e:
            verdict["reason"] = f"Parse error: {e} | Raw: {raw_response[:200]}"

    status = verdict.get("status", "FAILURE")
    headline = verdict.get("headline") or verdict.get("extracted_value", "")
    detailed_summary = verdict.get("detailed_summary") or headline
    reason = verdict.get("reason", "")

    payload = json.dumps(
        {
            "headline": headline,
            "detailed_summary": detailed_summary,
            "reason": reason,
        },
        ensure_ascii=True,
    )

    log_entry = f"{status}|{task_id}|{payload}"
    preview = headline or detailed_summary[:160]
    print(f"--- Reflection verdict: {status}|{task_id}|{preview}|Reason: {reason} ---")

    return {
        "reflexion_scratchpad": [log_entry]
    }
