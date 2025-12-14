from __future__ import annotations
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # OpenAI chat message dict를 누적
    messages: Annotated[List[Dict[str, Any]], add_messages]

    # LLM이 요청한 tool_calls를 임시 저장
    tool_calls: Optional[List[Dict[str, Any]]]

    steps: int
