from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.app.graph.state import AgentState
from src.app.graph.nodes import llm_node, tool_node


def route_after_llm(state: AgentState):
    # 1️⃣ step 제한 (안전장치)
    if state.get("steps", 0) >= 6:
        return END

    messages = state.get("messages", [])
    if not messages:
        return END

    last = messages[-1]

    # 2️⃣ 마지막 메시지가 tool 결과면 종료
    if isinstance(last, dict):
        if last.get("role") == "tool":
            return END
    else:
        # LangChain / LangGraph Message 객체
        role = getattr(last, "role", None)
        if role == "tool":
            return END

    # 3️⃣ tool_calls 있을 때만 tool로
    if state.get("tool_calls"):
        return "tool"

    return END




def build_app(enable_interrupt: bool = False):
    g = StateGraph(AgentState)

    g.add_node("llm", llm_node)
    g.add_node("tool", tool_node)

    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", route_after_llm)
    g.add_edge("tool", "llm")

    checkpointer = MemorySaver()

    if enable_interrupt:
        return g.compile(
            checkpointer=checkpointer,
            interrupt_before=["tool"],
        )

    return g.compile(checkpointer=checkpointer)
