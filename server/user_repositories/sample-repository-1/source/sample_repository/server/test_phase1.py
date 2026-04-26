import os
import json
from dotenv import load_dotenv

# Load environment variables before importing anything else!
load_dotenv()

from src.graph.workflow import build_synapse_graph
from src.tools.tool_selector import execution_warnings

def run_agent(prompt: str) -> dict:
    graph = build_synapse_graph()
    result = graph.invoke({"user_prompt": prompt, "loop_count": 0, "reflexion_scratchpad": []})

    # Surface any warnings that accumulated during execution
    if execution_warnings:
        print("\n" + "═" * 60)
        print("EXECUTION WARNINGS")
        print("═" * 60)
        for w in execution_warnings:
            if w["tool_chosen"]:
                print(f"\n  Task  : {w['task_id']}")
                print(f"  For task with description : \"{w['original']}\"")
                print(f"  No appropriate tool was found and hence the prompt")
                print(f"  was simplified to this    : \"{w['simplified']}\"")
                print(f"  Tool used : {w['tool_chosen']}")
            else:
                print(f"\n  Task  : {w['task_id']}")
                print(f"  For task with description : \"{w['original']}\"")
                print(f"  No appropriate tool was found after "
                      f"{w['steps_taken']} simplification attempts.")
                print(f"  This task was skipped.")
        print("═" * 60 + "\n")

    return result
def test_generation_loop():
    print("Initializing SynapseAI Phase 1 Test...\n")
    
    # 1. Compile the Orchestrator Graph
    app = build_synapse_graph()
    
    # 2. Define the starting state with a test prompt
    initial_state = {
        "user_prompt": "Create a 100 step DAG in which every event is a decision a person should make every year in their life. For example the step_1 would be a decision they make when they are 1 years old, the decision should be apt for their age. The person wants to become a competitive programmer and then a full time dancer and then retire to maintain his garden",
        "current_dag": {},
        "judge_score": 0.0,
        "feedback": "",
        "loop_count": 0,
        "current_step_id": "",
        "reflexion_scratchpad": [],
        "final_output": ""
    }
    
    print(f"Target Objective: {initial_state['user_prompt']}\n")
    print("-" * 50)
    
    # 3. Invoke the graph (This starts the Groq -> Gemini -> Groq loop)
    final_state = app.invoke(initial_state)
    
    # 4. Print the final results
    print("\n" + "=" * 50)
    print("🏁 PHASE 1 COMPLETE: FINAL APPROVED DAG")
    print("=" * 50)
    print(f"Total Generation Attempts: {final_state['loop_count']}")
    print(f"Final Judge Score: {final_state['judge_score']}/15")
    print("\nFinal DAG JSON:")
    print(json.dumps(final_state["current_dag"], indent=2))

if __name__ == "__main__":
    test_generation_loop()