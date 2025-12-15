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


# =====================================================
# ✅ LLM 이후 라우팅 (수정 핵심)
# =====================================================
def route_after_llm(state: AgentState):
    """
    LLM 실행 후:
    - tool_calls 있으면 → tool
    - 없으면 → reflection
    - step 제한 초과 시에도 → reflection
    """
    if state.get("steps", 0) >= 8:
        return "reflection"

    return "tool" if state.get("tool_calls") else "reflection"


def build_app(enable_interrupt: bool = False):
    g = StateGraph(AgentState)

    # =========================
    # 노드 등록
    # =========================
    g.add_node("memory_read", memory_read_node)
    g.add_node("llm", llm_node)
    g.add_node("tool", tool_node)
    g.add_node("reflection", reflection_node)

    # =========================
    # 그래프 흐름 (정답 구조)
    # =========================
    g.add_edge(START, "memory_read")
    g.add_edge("memory_read", "llm")

    # 🔥 핵심: llm 다음은 반드시 tool 또는 reflection 중 하나
    g.add_conditional_edges("llm", route_after_llm)

    # tool 실행 후 다시 llm
    g.add_edge("tool", "llm")

    # reflection 이후 종료
    g.add_edge("reflection", END)

    # =========================
    # 체크포인터
    # =========================
    checkpointer = MemorySaver()

    if enable_interrupt:
        return g.compile(
            checkpointer=checkpointer,
            interrupt_before=["tool"],
        )

    return g.compile(checkpointer=checkpointer)
