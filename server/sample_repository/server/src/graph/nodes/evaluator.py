import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.graph.state import SynapseState
from src.config.prompts import EVALUATOR_SYSTEM_PROMPT
from src.config.settings import GROQ_API_KEY_EVALUATOR

def check_for_cycles(tasks: list) -> bool:
    """
    Returns True if a cycle is detected in the task dependencies, False otherwise.
    Uses Depth-First Search (DFS).
    """
    graph = {task["id"]: task.get("dependencies", []) for task in tasks}
    visited = set()
    rec_stack = set()

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True 
                
        rec_stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False

def evaluate_dag(state: SynapseState) -> dict:
    """
    Phase 1: Evaluates the DAG based on a custom 15-point metric using Groq.
    """
    print("--- EVALUATING DAG (Custom Metric via Groq) ---")
    
    dag = state.get("current_dag", {})
    tasks = dag.get("tasks", [])
    user_prompt = state.get("user_prompt", "")
    
    total_score = 0
    feedback_notes = []
    
    # --- METRIC 1: Cycle Detection (5 Marks) ---
    if not tasks:
        feedback_notes.append("CRITICAL: No tasks generated or JSON formatting failed.")
    elif check_for_cycles(tasks):
        feedback_notes.append("CRITICAL: Cycle detected in DAG dependencies.")
    else:
        total_score += 5  
        
    # --- METRICS 2 & 3: Relevance & Dependencies via Groq (10 Marks) ---
    judge_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0, # Keep this at 0.0 so the judge is strict and deterministic
        api_key=GROQ_API_KEY_EVALUATOR 
    )
    
    user_content = f"Objective: {user_prompt}\n\nProposed DAG:\n{json.dumps(dag, indent=2)}"
    
    messages = [
        SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
        HumanMessage(content=user_content)
    ]
    
    response = judge_llm.invoke(messages)
    raw_response = response.content.strip()
    
    # Extract JSON from Groq's response
    start_idx = raw_response.find('{')
    end_idx = raw_response.rfind('}')
    
    if start_idx != -1 and end_idx != -1:
        try:
            eval_json = json.loads(raw_response[start_idx:end_idx+1])
            
            relevance_score = eval_json.get("relevance_score", 0)
            dependency_score = eval_json.get("dependency_score", 0)
            llm_feedback = eval_json.get("feedback", "No feedback provided.")
            
            total_score += (relevance_score + dependency_score)
            
            if llm_feedback != "APPROVED":
                feedback_notes.append(f"Logic Feedback: {llm_feedback}")
                
        except json.JSONDecodeError:
            feedback_notes.append("Judge evaluation failed to parse correctly.")
    else:
        feedback_notes.append("Judge evaluation failed to return JSON.")
    
    final_feedback = "\n".join(feedback_notes) if feedback_notes else "APPROVED"

    return {
        "judge_score": total_score,
        "feedback": final_feedback
    }