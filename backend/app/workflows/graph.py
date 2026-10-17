"""Build and compile the Twin LangGraph workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.workflows.nodes import WorkflowDeps, make_nodes
from app.workflows.routing import after_assess, after_plan, after_validate
from app.workflows.state import TwinGraphState


def build_twin_graph(deps: WorkflowDeps, *, checkpointer=None):
    nodes = make_nodes(deps)
    graph = StateGraph(TwinGraphState)

    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "load_conversation_context")
    graph.add_edge("load_conversation_context", "load_conversation_summary")
    graph.add_edge("load_conversation_summary", "retrieve_relevant_memories")
    graph.add_edge("retrieve_relevant_memories", "resolve_contextual_query")
    graph.add_edge("resolve_contextual_query", "plan_query")
    graph.add_conditional_edges("plan_query", after_plan)
    graph.add_edge("parallel_retrieval", "merge_evidence")
    graph.add_edge("merge_evidence", "assess_evidence")
    graph.add_conditional_edges("assess_evidence", after_assess)
    graph.add_edge("plan_retrieval_retry", "parallel_retrieval")
    graph.add_edge("generate_answer", "validate_answer")
    graph.add_conditional_edges("validate_answer", after_validate)
    graph.add_edge("repair_answer", "validate_answer")
    for terminal in (
        "finalize_success",
        "finalize_abstain",
        "finalize_clarification",
        "finalize_error",
        "finalize_safe_fallback",
    ):
        graph.add_edge(terminal, END)

    return graph.compile(checkpointer=checkpointer)


def build_playground_graph(deps: WorkflowDeps, *, checkpointer=None):
    """Playground workflow: no conversation memory/summary nodes."""
    nodes = make_nodes(deps)
    graph = StateGraph(TwinGraphState)

    playground_nodes = {
        k: v
        for k, v in nodes.items()
        if k
        not in {
            "load_conversation_context",
            "load_conversation_summary",
            "retrieve_relevant_memories",
        }
    }
    for name, fn in playground_nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "resolve_contextual_query")
    graph.add_edge("resolve_contextual_query", "plan_query")
    graph.add_conditional_edges("plan_query", after_plan)
    graph.add_edge("parallel_retrieval", "merge_evidence")
    graph.add_edge("merge_evidence", "assess_evidence")
    graph.add_conditional_edges("assess_evidence", after_assess)
    graph.add_edge("plan_retrieval_retry", "parallel_retrieval")
    graph.add_edge("generate_answer", "validate_answer")
    graph.add_conditional_edges("validate_answer", after_validate)
    graph.add_edge("repair_answer", "validate_answer")
    for terminal in (
        "finalize_success",
        "finalize_abstain",
        "finalize_clarification",
        "finalize_error",
        "finalize_safe_fallback",
    ):
        graph.add_edge(terminal, END)

    return graph.compile(checkpointer=checkpointer)
