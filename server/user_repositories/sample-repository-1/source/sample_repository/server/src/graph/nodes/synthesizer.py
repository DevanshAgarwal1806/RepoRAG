import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.graph.state import SynapseState
from src.config.keypool import pool


def synthesize_output(state: SynapseState) -> dict:
    print("--- SYNTHESIZING FINAL OUTPUT ---")

    scratchpad  = state.get("reflexion_scratchpad", [])
    user_prompt = state.get("user_prompt", "")
    dag         = state.get("current_dag", {})

    # Collect all successful results, labelled by their task description
    task_lookup = {
        task["id"]: task.get("description", task["id"])
        for task in dag.get("tasks", [])
    }
    raw_results = _collect_latest_raw_results(scratchpad)

    successful_results = []
    for entry in scratchpad:
        if entry.startswith("SUCCESS"):
            parts     = entry.split("|", 2)
            task_id   = parts[1] if len(parts) > 1 else "unknown"
            payload   = _parse_success_payload(parts[2] if len(parts) > 2 else "")
            task_desc = task_lookup.get(task_id, task_id)
            successful_results.append(
                "\n".join([
                    f"[{task_desc}]",
                    f"Headline: {_truncate_text(payload['headline'], 220)}",
                    f"Detailed summary: {_truncate_text(payload['detailed_summary'], 900)}",
                    f"Reflection reason: {_truncate_text(payload['reason'], 180)}",
                    f"Raw evidence excerpt: {_truncate_text(raw_results.get(task_id, ''), 500)}",
                ])
            )

    if not successful_results:
        state['useful_output'] = False
        return {
            "final_output": (
                "The agent was unable to complete any tasks successfully. "
                "Please check the warnings above and try again."
            )
        }

    # Build the context block for the LLM
    gathered_context = "\n\n---\n\n".join(successful_results)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=pool.next()
    )

    messages = [
        SystemMessage(content=(
            "You are an expert research analyst. "
            "You have been given raw research results gathered by an AI agent on behalf of a user. "
            "Your job is to transform this raw research into a comprehensive, authoritative response "
            "that directly answers the user's question.\n\n"
            "STRICT RULES:\n"
            "1. NEVER mention research, links, sources, papers, articles, agents, or tools. "
            "   Write as if you already knew all of this information.\n"
            "2. NEVER say phrases like 'the research shows', 'according to the results', "
            "   'the links indicate', 'based on the gathered data', or anything similar.\n"
            "3. Present ALL useful facts, numbers, names, dates, and details directly "
            "   as statements of fact.\n"
            "4. Structure your response with clear headers and paragraphs. "
            "   Use headers to organize topics, paragraphs to explain them.\n"
            "5. If the results contain specific numbers, prices, names, or dates — include them. "
            "   Do not vaguely summarize when specific information is available.\n"
            "6. Write in second or third person as appropriate to the question. "
            "   Never first person ('I found...', 'We gathered...').\n"
            "7. Prioritize depth and usefulness. Give concrete recommendations, practical tradeoffs, "
            "   day-by-day specifics, estimated ranges, and logistics when the context supports them.\n"
            "8. Synthesize across all task results into a complete answer rather than listing isolated facts.\n"
            "9. Do not pad the response. Do not add conclusions like 'I hope this helps'. "
            "   End when the information is complete."
        )),
        HumanMessage(content=(
            f"User's original question:\n{user_prompt}\n\n"
            f"Research results gathered by the agent:\n\n{gathered_context}"
        ))
    ]

    response = llm.invoke(messages)
    final    = response.content.strip()

    return {"final_output": final}


def _parse_success_payload(payload: str) -> dict[str, str]:
    fallback = {
        "headline": payload,
        "detailed_summary": payload,
        "reason": "",
    }

    if not payload:
        return fallback

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return fallback

    return {
        "headline": parsed.get("headline", ""),
        "detailed_summary": parsed.get("detailed_summary", "") or parsed.get("headline", ""),
        "reason": parsed.get("reason", ""),
    }


def _collect_latest_raw_results(scratchpad: list[str]) -> dict[str, str]:
    raw_results: dict[str, str] = {}

    for entry in scratchpad:
        if not entry.startswith("RAW_RESULT|"):
            continue

        parts = entry.split("|", 2)
        if len(parts) < 3:
            continue

        raw_results[parts[1]] = parts[2]

    return raw_results


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
