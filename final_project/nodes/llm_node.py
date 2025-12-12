from __future__ import annotations
from typing import Dict, Any
from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage

from state import State
from tools import register_default_tools

client = OpenAI()
_tool_registry = register_default_tools()
_tools_for_openai = _tool_registry.list_openai_tools()


def lc_to_openai_messages(messages):
    """LangChain Message → OpenAI dict 메시지"""
    out = []
    for m in messages:
        if isinstance(m, dict):
            out.append(m)
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": m.content})
    return out


def llm_node(state: State) -> Dict[str, Any]:
    messages = state["messages"]

    # 🔥 핵심 변환
    openai_messages = lc_to_openai_messages(messages)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=openai_messages,
        tools=_tools_for_openai,
        tool_choice="auto",
    )

    msg = resp.choices[0].message

    assistant_msg = {
        "role": "assistant",
        "content": msg.content,
    }

    if msg.tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            }
            for tc in msg.tool_calls
        ]

    return {
        "messages": messages + [assistant_msg]
    }
