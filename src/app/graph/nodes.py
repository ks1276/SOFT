from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.app.llm.client import chat_raw
from src.app.tools.__base__ import registry

# ⚠️ 중요: @tool 데코레이터가 import 시점에 registry 등록을 수행하므로 반드시 import
from src.app.tools import basic  # noqa: F401
from src.app.tools import rag_tools  # noqa: F401
from src.app.tools import memory_tools  # noqa: F401


# -----------------------------
# message/toolcall normalization
# -----------------------------
def _message_obj_to_dict(m: Any) -> Optional[Dict[str, Any]]:
    """
    LangChain/BaseMessage/HumanMessage/AIMessage/ToolMessage 등 객체를 dict로 변환.
    실패 시 None.
    """
    if isinstance(m, dict):
        return m
    if hasattr(m, "model_dump"):
        d = m.model_dump()
        return d if isinstance(d, dict) else None
    if hasattr(m, "dict"):
        d = m.dict()
        return d if isinstance(d, dict) else None
    return None


def _toolcall_obj_to_dict(tc: Any) -> Dict[str, Any]:
    """
    tool_calls 원소가 dict/객체 어떤 형태든 dict로 변환.
    """
    if isinstance(tc, dict):
        return tc
    if hasattr(tc, "model_dump"):
        d = tc.model_dump()
        return d if isinstance(d, dict) else {"value": d}
    if hasattr(tc, "dict"):
        d = tc.dict()
        return d if isinstance(d, dict) else {"value": d}
    return {"value": str(tc)}


def _normalize_messages(messages: Any) -> List[Dict[str, Any]]:
    """
    state["messages"]가 dict/객체가 섞여 있어도 OpenAI dict messages로 정규화.
    LangChain dump 형태(type=human/ai/tool)를 role=user/assistant/tool로 매핑.
    """
    out: List[Dict[str, Any]] = []
    if not messages:
        return out
    if not isinstance(messages, list):
        messages = [messages]

    for m in messages:
        # 1) dict 그대로
        if isinstance(m, dict):
            if "role" in m:
                out.append(m)
            else:
                out.append({"role": "user", "content": str(m)})
            continue

        # 2) 객체 -> dict
        d = _message_obj_to_dict(m)
        if isinstance(d, dict):
            t = d.get("type") or d.get("role")
            if t == "human":
                out.append({"role": "user", "content": d.get("content", "")})
            elif t == "ai":
                msg: Dict[str, Any] = {"role": "assistant", "content": d.get("content", "")}
                if d.get("tool_calls") is not None:
                    msg["tool_calls"] = d.get("tool_calls")
                out.append(msg)
            elif t == "tool":
                msg: Dict[str, Any] = {"role": "tool", "content": d.get("content", "")}
                if "tool_call_id" in d:
                    msg["tool_call_id"] = d["tool_call_id"]
                if "name" in d:
                    msg["name"] = d["name"]
                out.append(msg)
            else:
                # fallback
                out.append({"role": "user", "content": str(d)})
            continue

        # 3) (role, content) 튜플 방어
        if isinstance(m, tuple) and len(m) == 2:
            role, content = m
            out.append({"role": str(role), "content": str(content)})
            continue

        # 4) 최후 fallback
        out.append({"role": "user", "content": str(m)})

    return out


def _normalize_one_tool_call(tc_any: Any) -> Dict[str, Any]:
    """
    어떤 형태의 tool_call이 와도 OpenAI Chat Completions 규격으로 강제 변환:
    {
      "id": "...",
      "type": "function",
      "function": {"name": "...", "arguments": "{...json...}"}
    }
    """
    tc = dict(_toolcall_obj_to_dict(tc_any))

    # LangChain 스타일: {"name":..., "args":...}
    name = tc.get("name")
    args = tc.get("args")

    # OpenAI/혹은 섞인 스타일: {"function": {"name":..., "arguments":...}} 또는 {"function": {"name":..., "args":...}}
    fn = tc.get("function")
    if isinstance(fn, dict):
        name = fn.get("name", name)
        args = fn.get("arguments", fn.get("args", fn.get("input", args)))

    # 또 다른 케이스: top-level "arguments"
    if args is None:
        args = tc.get("arguments")

    # arguments는 반드시 JSON string
    if isinstance(args, dict):
        arguments = json.dumps(args, ensure_ascii=False)
    elif args is None:
        arguments = "{}"
    else:
        arguments = str(args)

    if not name:
        name = ""

    # id 없으면 생성(매칭용)
    if not tc.get("id"):
        tc["id"] = f"tc_{abs(hash(name + arguments))}"

    tc["type"] = "function"
    tc["function"] = {"name": name, "arguments": arguments}
    return tc


def _sanitize_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    OpenAI로 보내기 직전에 messages를 정리:
    - assistant.tool_calls 원소(dict/객체 섞임)를 전부 OpenAI 규격으로 변환
    """
    fixed: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue

        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            m = {"role": "user", "content": str(m)}
            fixed.append(m)
            continue

        if role == "assistant" and m.get("tool_calls"):
            tcs = m.get("tool_calls")
            if isinstance(tcs, list):
                m["tool_calls"] = [_normalize_one_tool_call(tc) for tc in tcs]

        fixed.append(m)
    return fixed


def _to_message_dict(resp: Any) -> Dict[str, Any]:
    """
    chat_raw()의 반환값이 어떤 형태든 "assistant message dict"로 정규화.
    """
    # dict인 경우
    if isinstance(resp, dict):
        if "choices" in resp and resp["choices"]:
            msg = resp["choices"][0].get("message")
            if isinstance(msg, dict):
                return msg
        if resp.get("role") in ("assistant", "tool", "user", "system"):
            return resp
        return resp

    # OpenAI/Pydantic 객체면 model_dump
    if hasattr(resp, "model_dump"):
        dumped = resp.model_dump()
        if isinstance(dumped, dict):
            if "choices" in dumped and dumped["choices"]:
                msg = dumped["choices"][0].get("message")
                if isinstance(msg, dict):
                    return msg
            return dumped

    return {"role": "assistant", "content": str(resp)}


# -----------------------------
# LangGraph nodes
# -----------------------------
def llm_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # 1) 상태 메시지 정규화(객체/딕셔너리 섞여도 OK)
    messages = _normalize_messages(state.get("messages"))

    # 2) OpenAI 규격으로 sanitize (tool_calls 강제 보정)
    messages = _sanitize_openai_messages(messages)

    tools = registry.list_openai_tools()

    resp = chat_raw(
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    msg = _to_message_dict(resp)

    # (중요) 응답 assistant msg에 tool_calls가 있다면 이것도 규격화해서 state에 넣기
    if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list):
            msg["tool_calls"] = [_normalize_one_tool_call(tc) for tc in tcs]

    out: Dict[str, Any] = {
        "messages": [msg],
        "tool_calls": msg.get("tool_calls") or None,
    }

    # 무한루프 방지용 step 카운터
    out["steps"] = int(state.get("steps", 0)) + 1
    return out


def tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    - llm_node가 만든 tool_calls를 실행
    - 결과를 role="tool" 메시지로 messages에 append
    - tool_calls는 처리했으니 None으로 비움
    """
    tool_calls_any = state.get("tool_calls") or None
    if not tool_calls_any:
        return {"tool_calls": None}

    # tool_calls도 혹시 객체 섞였을 수 있으니 정규화
    tool_calls: List[Dict[str, Any]] = []
    if isinstance(tool_calls_any, list):
        tool_calls = [_normalize_one_tool_call(tc) for tc in tool_calls_any]
    else:
        tool_calls = [_normalize_one_tool_call(tool_calls_any)]

    tool_messages: List[Dict[str, Any]] = []

    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name")
        arguments = fn.get("arguments", "{}")
        tool_call_id = tc.get("id") or ""

        if not name:
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": "",
                "content": "Tool call missing function name.",
            })
            continue

        try:
            result = registry.invoke(name, arguments)
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except Exception as e:
            content = f"[tool_error] {type(e).__name__}: {e}"

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        })

    return {
        "messages": tool_messages,
        "tool_calls": None,
    }
