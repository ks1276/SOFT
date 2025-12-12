from __future__ import annotations
from typing import Dict, Any
import json

from state import State
from tools import register_default_tools

_tool_registry = register_default_tools()


def tool_node(state: State) -> Dict[str, Any]:
    messages = state["messages"]
    last = messages[-1]

    # -----------------------
    # dict / AIMessage 모두 대응
    # -----------------------
    tool_calls = None

    if isinstance(last, dict):
        tool_calls = last.get("tool_calls")
    else:
        tool_calls = getattr(last, "tool_calls", None)

    if not tool_calls:
        return {}

    tool_messages = []
    last_result = None

    for tc in tool_calls:
        # OpenAI tool-call 형식
        func = tc.get("function", {})
        name = func.get("name")
        args_str = func.get("arguments") or "{}"

        try:
            args = json.loads(args_str)
        except Exception:
            args = {}

        result = _tool_registry.call(name, args)
        last_result = result

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": name,  # 🔥 중요
            "content": json.dumps(result, ensure_ascii=False),
        })

    return {
        # 🔥 messages 누적 필수
        "messages": messages + tool_messages,
        "tool_result": last_result,
    }
