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
    out: List[Dict[str, Any]] = []
    if not messages:
        return out
    if not isinstance(messages, list):
        messages = [messages]

    for m in messages:
        if isinstance(m, dict):
            if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list) and not m["tool_calls"]:
                m = dict(m)
                m.pop("tool_calls", None)
            out.append(m)
            continue

        d = _message_obj_to_dict(m)
        if isinstance(d, dict):
            t = d.get("type") or d.get("role")
            if t == "human":
                out.append({"role": "user", "content": d.get("content", "")})
            elif t == "ai":
                msg = {"role": "assistant", "content": d.get("content", "")}
                if d.get("tool_calls"):
                    msg["tool_calls"] = d["tool_calls"]
                out.append(msg)
            elif t == "tool":
                msg = {"role": "tool", "content": d.get("content", "")}
                if "tool_call_id" in d:
                    msg["tool_call_id"] = d["tool_call_id"]
                if "name" in d:
                    msg["name"] = d["name"]
                out.append(msg)
            else:
                out.append({"role": "user", "content": str(d)})
            continue

        out.append({"role": "user", "content": str(m)})

    return out


def _normalize_one_tool_call(tc_any: Any) -> Dict[str, Any]:
    tc = dict(_toolcall_obj_to_dict(tc_any))

    name = tc.get("name")
    args = tc.get("args")

    fn = tc.get("function")
    if isinstance(fn, dict):
        name = fn.get("name", name)
        args = fn.get("arguments", fn.get("args", args))

    if args is None:
        args = tc.get("arguments")

    if isinstance(args, dict):
        arguments = json.dumps(args, ensure_ascii=False)
    elif args is None:
        arguments = "{}"
    else:
        arguments = str(args)

    if not tc.get("id"):
        tc["id"] = f"tc_{abs(hash((name or '') + arguments))}"

    tc["type"] = "function"
    tc["function"] = {"name": name or "", "arguments": arguments}
    return tc


def _sanitize_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fixed: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue

        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            fixed.append({"role": "user", "content": str(m)})
            continue

        if role == "assistant" and isinstance(m.get("tool_calls"), list) and not m["tool_calls"]:
            m = dict(m)
            m.pop("tool_calls", None)

        if role == "assistant" and m.get("tool_calls"):
            m = dict(m)
            m["tool_calls"] = [_normalize_one_tool_call(tc) for tc in m["tool_calls"]]

        fixed.append(m)

    return fixed


def _to_message_dict(resp: Any) -> Dict[str, Any]:
    if isinstance(resp, dict):
        if "choices" in resp and resp["choices"]:
            msg = resp["choices"][0].get("message")
            if isinstance(msg, dict):
                return msg
        return resp
    if hasattr(resp, "model_dump"):
        d = resp.model_dump()
        if isinstance(d, dict) and d.get("choices"):
            return d["choices"][0]["message"]
    return {"role": "assistant", "content": str(resp)}


# -----------------------------
# LangGraph nodes
# -----------------------------
def llm_node(state: Dict[str, Any]) -> Dict[str, Any]:
    messages = _sanitize_openai_messages(
        _normalize_messages(state.get("messages"))
    )

    tools = registry.list_openai_tools()

    # 🔥 핵심: RAG 한 번 사용했으면 tool 다시 못 쓰게
    tool_choice = "none" if state.get("rag_used") else "auto"

    resp = chat_raw(
        messages=messages,
        tools=tools if tool_choice == "auto" else None,  # 🔥 중요
        tool_choice=tool_choice,
    )

    msg = _to_message_dict(resp)

    if isinstance(msg, dict) and msg.get("role") == "assistant":
        if isinstance(msg.get("tool_calls"), list):
            if not msg["tool_calls"]:
                msg.pop("tool_calls", None)
            else:
                msg["tool_calls"] = [
                    _normalize_one_tool_call(tc)
                    for tc in msg["tool_calls"]
                ]

    return {
        "messages": [msg],
        "tool_calls": msg.get("tool_calls"),
        "steps": int(state.get("steps", 0)) + 1,

        # 🔥 유지
        "rag_used": state.get("rag_used", False),
    }


def tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    tool_calls_any = state.get("tool_calls")
    if not tool_calls_any:
        return {}

    tool_calls = (
        [_normalize_one_tool_call(tc) for tc in tool_calls_any]
        if isinstance(tool_calls_any, list)
        else [_normalize_one_tool_call(tool_calls_any)]
    )

    tool_messages: List[Dict[str, Any]] = []
    rag_used = state.get("rag_used", False)

    for tc in tool_calls:
        fn = tc["function"]
        name = fn["name"]
        arguments = fn.get("arguments", "{}")
        tool_call_id = tc["id"]

        try:
            result = registry.invoke(name, arguments)
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except Exception as e:
            content = f"[tool_error] {e}"

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        })

        # 🔥 핵심: rag_search가 실행되면 플래그 ON
        if name.startswith("rag"):
            rag_used = True

    return {
        "messages": tool_messages,
        "tool_calls": None,
        "rag_used": rag_used,
    }
