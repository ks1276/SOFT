from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.app.graph.state import AgentState
from src.app.graph.nodes import llm_node, tool_node


def route_after_llm(state: AgentState):
    if state.get("steps", 0) >= 8:
        return END
    return "tool" if state.get("tool_calls") else END


def build_app(enable_interrupt: bool = False):
    g = StateGraph(AgentState)
    g.add_node("llm", llm_node)
    g.add_node("tool", tool_node)

    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", route_after_llm)
    g.add_edge("tool", "llm")

    checkpointer = MemorySaver()

    if enable_interrupt:
        return g.compile(checkpointer=checkpointer, interrupt_before=["tool"])
    return g.compile(checkpointer=checkpointer)
