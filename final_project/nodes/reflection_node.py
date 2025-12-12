from __future__ import annotations
from typing import Dict, Any, List

from state import State


def normalize_message_content(msg):
    """
    dict / LangChain Message 모두에서 content 추출
    """
    # dict
    if isinstance(msg, dict):
        return msg.get("content")

    # LangChain Message
    return getattr(msg, "content", None)


def reflection_node(state: State) -> Dict[str, Any]:
    messages = state["messages"]

    # 최근 메시지 몇 개만 요약/반영
    recent = messages[-5:]

    contents = [
        normalize_message_content(m)
        for m in recent
        if normalize_message_content(m)
    ]

    if not contents:
        return {}

    reflection_msg = {
        "role": "assistant",
        "content": "🪞 Reflection:\n" + "\n".join(contents),
    }

    return {
        "messages": messages + [reflection_msg]
    }
