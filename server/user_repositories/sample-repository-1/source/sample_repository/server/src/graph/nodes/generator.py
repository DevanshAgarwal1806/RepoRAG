
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.graph.state import SynapseState
from src.config.prompts import GENERATOR_SYSTEM_PROMPT
from src.config.settings import GROQ_API_KEY_GENERATOR
from src.tools.registry import build_tool_manifest

def generate_dag(state: SynapseState) -> dict:
    print("--- GENERATING DAG (Groq) ---")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=GROQ_API_KEY_GENERATOR
    )

    user_prompt = state.get("user_prompt", "")
    feedback = state.get("feedback", "")
    loop_count = state.get("loop_count", 0)
    judge_score = state.get("judge_score", 0)

    # Build the system prompt with the live tool manifest injected
    tool_manifest = build_tool_manifest()
    system_prompt = GENERATOR_SYSTEM_PROMPT.format(tool_manifest=tool_manifest)

    user_content = f"Target Objective: {user_prompt}"

    if feedback and loop_count > 0:
        user_content += (
            f"\n\nCRITICAL: Your previous DAG was rejected with a score of "
            f"{judge_score}/15. Judge's feedback:\n{feedback}\n\n"
            f"Output an updated JSON DAG that fixes these issues."
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]

    response = llm.invoke(messages)
    raw_output = response.content.strip()

    start_idx = raw_output.find('{')
    end_idx = raw_output.rfind('}')

    if start_idx != -1 and end_idx != -1:
        try:
            dag_json = json.loads(raw_output[start_idx:end_idx + 1])
        except json.JSONDecodeError:
            dag_json = {"tasks": [], "error": "Extracted string was not valid JSON."}
    else:
        dag_json = {"tasks": [], "error": "No JSON brackets found in output."}

    return {
        "current_dag": dag_json,
        "loop_count": loop_count + 1
    }