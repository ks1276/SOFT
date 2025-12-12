from __future__ import annotations
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph.message import add_messages


class State(TypedDict):
    """
    LangGraph State
    - messages: Annotated list → LangGraph reducer add_messages 사용
    - tool_result: 마지막 tool 실행 결과
    - memory_context: long-term memory에서 가져온 문장들
    """
    messages: Annotated[List[Dict[str, Any]], add_messages]
    tool_result: Optional[Dict[str, Any]]
    memory_context: List[str]
