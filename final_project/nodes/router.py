from state import State

def route_after_llm(state: State) -> str:
    messages = state["messages"]
    last = messages[-1]

    # dict 메시지
    if isinstance(last, dict):
        if last.get("tool_calls"):
            return "tools"
        return "reflection"

    # LangChain AIMessage
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return "tools"

    return "reflection"
