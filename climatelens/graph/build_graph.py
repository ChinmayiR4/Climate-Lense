"""
Wires the node functions into a LangGraph StateGraph.

Graph shape:

    retrieve --> grade_relevance --(relevance_ok=False, attempts<2)--> retrieve
                       |
                       v (relevance_ok=True, or attempts exhausted)
                compare_to_benchmark
                       |
                       v
                generate_answer
                       |
                       v
                log_interaction --> END
"""

import time

from langgraph.graph import StateGraph, END

from graph.state import ClimateLensState
from graph.nodes import (
    retrieve,
    grade_relevance,
    should_retry_retrieval,
    compare_to_benchmark,
    generate_answer,
    log_interaction,
)


def build_graph():
    graph = StateGraph(ClimateLensState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_relevance", grade_relevance)
    graph.add_node("compare_to_benchmark", compare_to_benchmark)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("log_interaction", log_interaction)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_relevance")

    graph.add_conditional_edges(
        "grade_relevance",
        should_retry_retrieval,
        {"retrieve": "retrieve", "next": "compare_to_benchmark"},
    )

    graph.add_edge("compare_to_benchmark", "generate_answer")
    graph.add_edge("generate_answer", "log_interaction")
    graph.add_edge("log_interaction", END)

    return graph.compile()


def run_query(session_id: str, query: str, target_doc_type: str | None = None) -> ClimateLensState:
    app = build_graph()
    initial_state: ClimateLensState = {
        "session_id": session_id,
        "query": query,
        "target_doc_type": target_doc_type,
        "retrieval_attempts": 0,
        "start_time": time.time(),
    }
    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    result = run_query(
        session_id="test-session-1",
        query="Does UK's NDC pledge align with the IPCC 1.5C pathway on emissions reduction?",
        target_doc_type="ndc_pledge",
    )
    print("ANSWER:\n", result["answer"])
    print("\nALIGNMENT SCORE:", result.get("alignment_score"))
    print("FLAGS:", result.get("flags"))
