from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import State
from nodes.llm_node import llm_node
from nodes.tool_node import tool_node
from nodes.router import route_after_llm
from nodes.reflection_node import reflection_node

# ✔ thread_id 기반 메모리 유지
memory = MemorySaver()

workflow = StateGraph(State)

# -----------------------
# Nodes
# -----------------------
workflow.add_node("llm", llm_node)
workflow.add_node("tools", tool_node)
workflow.add_node("reflection", reflection_node)

# -----------------------
# Entry
# -----------------------
workflow.set_entry_point("llm")

# -----------------------
# Edges
# -----------------------
workflow.add_edge("tools", "llm")        # tool → 다시 LLM
workflow.add_edge("reflection", END)     # reflection → 종료

workflow.add_conditional_edges(
    "llm",
    route_after_llm,
    {
        "tools": "tools",
        "reflection": "reflection",
    }
)

# -----------------------
# Compile
# -----------------------
graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["tools"],   # tool 전에 interrupt
)
