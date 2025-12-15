from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.app.graph.state import AgentState
from src.app.graph.nodes import (
    llm_node,
    tool_node,
    memory_read_node,
    reflection_node,
)


def route_after_llm(state: AgentState):
    if state.get("steps", 0) >= 8:
        return END
    return "tool" if state.get("tool_calls") else END


def build_app(enable_interrupt: bool = False):
    g = StateGraph(AgentState)

    # 노드 등록
    g.add_node("memory_read", memory_read_node)
    g.add_node("llm", llm_node)
    g.add_node("tool", tool_node)
    g.add_node("reflection", reflection_node)

    # 흐름
    g.add_edge(START, "memory_read")
    g.add_edge("memory_read", "llm")

    g.add_conditional_edges("llm", route_after_llm)
    g.add_edge("tool", "llm")

    g.add_edge("llm", "reflection")
    g.add_edge("reflection", END)

    checkpointer = MemorySaver()

    if enable_interrupt:
        return g.compile(
            checkpointer=checkpointer,
            interrupt_before=["tool"],
        )

    return g.compile(checkpointer=checkpointer)
