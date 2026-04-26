from langgraph.graph import StateGraph, END
from src.graph.state import SynapseState
from src.graph.nodes.generator import generate_dag
from src.graph.nodes.evaluator import evaluate_dag
from src.graph.nodes.executor import execute_task
from src.graph.nodes.reflector import reflect_on_execution
from src.graph.nodes.synthesizer import synthesize_output
from src.graph.edges.routers import route_evaluation, route_execution


def build_synapse_graph():
    builder = StateGraph(SynapseState)

    # Nodes
    builder.add_node("generator", generate_dag)
    builder.add_node("evaluator", evaluate_dag)
    builder.add_node("executor", execute_task)
    builder.add_node("reflector", reflect_on_execution)
    builder.add_node("synthesizer", synthesize_output)  

    # Entry point
    builder.set_entry_point("generator")

    # Phase 1 loop
    builder.add_edge("generator", "evaluator")
    builder.add_conditional_edges(
        "evaluator",
        route_evaluation,
        {
            "generate": "generator",
            "execute": "executor",
        }
    )

    # Phase 3 loop: executor → reflector → router → synthesizer → back or done
    builder.add_edge("executor", "reflector")
    builder.add_conditional_edges(
        "reflector",
        route_execution,
        {
            "next_task": "executor",   # success, still tasks left
            "retry":     "executor",   # failure, retry same task
            "done":      "synthesizer",          # all tasks complete
            "fail":      "synthesizer",          # task exhausted retries
        }
    )

    #Synthesizer always ends the DAG
    builder.add_edge("synthesizer",END)
    return builder.compile()