from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.app.llm.client import chat_raw
from src.app.tools.__base__ import registry

# ⚠️ 중요: @tool 데코레이터가 import 시점에 registry 등록을 수행하므로 반드시 import
from src.app.tools import basic  # noqa: F401
from src.app.tools import rag_tools  # noqa: F401
from src.app.tools import memory_tools  # noqa: F401


# =====================================================
# message / toolcall normalization (기존 코드 유지)
# =====================================================
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
            if "role" in m:
                if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list) and len(m["tool_calls"]) == 0:
                    m = dict(m)
                    m.pop("tool_calls", None)
                out.append(m)
            else:
                out.append({"role": "user", "content": str(m)})
            continue

        d = _message_obj_to_dict(m)
        if isinstance(d, dict):
            t = d.get("type") or d.get("role")
            if t == "human":
                out.append({"role": "user", "content": d.get("content", "")})
            elif t == "ai":
                msg: Dict[str, Any] = {"role": "assistant", "content": d.get("content", "")}
                if d.get("tool_calls"):
                    msg["tool_calls"] = d["tool_calls"]
                out.append(msg)
            elif t == "tool":
                msg2: Dict[str, Any] = {"role": "tool", "content": d.get("content", "")}
                if "tool_call_id" in d:
                    msg2["tool_call_id"] = d["tool_call_id"]
                if "name" in d:
                    msg2["name"] = d["name"]
                out.append(msg2)
            else:
                out.append({"role": "user", "content": str(d)})
            continue

        if isinstance(m, tuple) and len(m) == 2:
            role, content = m
            out.append({"role": str(role), "content": str(content)})
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

    if not name:
        name = ""

    if not tc.get("id"):
        tc["id"] = f"tc_{abs(hash(name + arguments))}"

    tc["type"] = "function"
    tc["function"] = {"name": name, "arguments": arguments}
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

        if role == "assistant" and isinstance(m.get("tool_calls"), list) and len(m["tool_calls"]) == 0:
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
        dumped = resp.model_dump()
        if isinstance(dumped, dict):
            if "choices" in dumped and dumped["choices"]:
                msg = dumped["choices"][0].get("message")
                if isinstance(msg, dict):
                    return msg
            return dumped

    return {"role": "assistant", "content": str(resp)}


# =====================================================
# LangGraph Nodes
# =====================================================

def llm_node(state: Dict[str, Any]) -> Dict[str, Any]:
    messages = _normalize_messages(state.get("messages"))
    messages = _sanitize_openai_messages(messages)

    resp = chat_raw(
        messages=messages,
        tools=registry.list_openai_tools(),
        tool_choice="auto",
    )
    msg = _to_message_dict(resp)

    if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
        if not msg["tool_calls"]:
            msg.pop("tool_calls", None)
        else:
            msg["tool_calls"] = [_normalize_one_tool_call(tc) for tc in msg["tool_calls"]]

    return {
        "messages": [msg],
        "tool_calls": msg.get("tool_calls"),
        "steps": int(state.get("steps", 0)) + 1,
    }


def tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    tool_calls_any = state.get("tool_calls")
    if not tool_calls_any:
        return {"tool_calls": None}

    tool_calls = [_normalize_one_tool_call(tc) for tc in tool_calls_any]
    tool_messages: List[Dict[str, Any]] = []

    for tc in tool_calls:
        fn = tc["function"]
        name = fn["name"]
        args = fn["arguments"]
        tcid = tc["id"]

        try:
            result = registry.invoke(name, args)
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except Exception as e:
            content = f"[tool_error] {e}"

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tcid,
            "name": name,
            "content": content,
        })

    return {"messages": tool_messages, "tool_calls": None}


# =====================================================
# ✅ 추가된 노드들 (이번 수정의 핵심)
# =====================================================

def memory_read_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("memory_checked"):
        return {}

    from src.app.memory.store import read_memory

    # ✅ 핵심: messages 정규화
    messages = _normalize_messages(state.get("messages", []))

    user_msg = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "",
    )

    if not user_msg:
        return {"memory_checked": True}

    items = read_memory(user_msg, top_k=3)
    if not items:
        return {"memory_checked": True}

    mem_text = "\n".join(f"- ({it.memory_type}) {it.content}" for it in items)

    return {
        "messages": [{
            "role": "system",
            "content": f"[RELATED MEMORY]\n{mem_text}",
        }],
        "memory_checked": True,
    }


def reflection_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    대화 종료 시점에서:
    - 최근 대화를 요약
    - 장기 메모리로 저장할 가치가 있는지 LLM으로 판단
    - 필요하면 memory DB에 write

    ⚠️ 설계 포인트
    - 첫 턴(step <= 1)에서는 reflection 생략 (속도 + UX)
    - state["messages"]는 반드시 정규화 후 사용
    - reflection 실패가 전체 그래프를 깨지 않도록 방어
    """

    # ✅ 1. 첫 턴에서는 reflection 하지 않음 (속도 개선 핵심)
    if state.get("steps", 0) <= 1:
        return {}

    try:
        # 지연 import (reflection 안 쓸 땐 비용 0)
        from src.app.memory.reflection import build_snippet, run_memory_extractor
        from src.app.memory.store import write_memory
        from src.app.config.settings import settings
        from langchain_openai import ChatOpenAI
    except Exception as e:
        # 환경 문제로 reflection이 불가능해도 전체 그래프는 계속
        return {
            "messages": [{
                "role": "system",
                "content": f"[reflection skipped: import error] {e}",
            }]
        }

    # ✅ 2. messages 정규화 (HumanMessage / dict 혼합 방지)
    messages = _normalize_messages(state.get("messages", []))

    # 최근 user / assistant 발화 추출
    user_msg = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    final_answer = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "assistant"),
        "",
    )

    if not user_msg or not final_answer:
        return {}

    # ✅ 3. extractor에 넘길 스니펫 구성
    snippet = build_snippet(
        history=messages,
        user_message=user_msg,
        final_answer=final_answer,
    )

    # ✅ 4. Reflection 판단용 LLM (temperature 0 고정)
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
    )

    try:
        result = run_memory_extractor(llm, snippet)
    except Exception as e:
        # reflection 판단 실패 → 조용히 skip
        return {
            "messages": [{
                "role": "system",
                "content": f"[reflection skipped: extractor error] {e}",
            }]
        }

    # ✅ 5. 저장할 가치 없으면 종료
    if not result.get("should_write_memory"):
        return {}

    # ✅ 6. Memory write
    try:
        write_memory(
            content=result["content"],
            memory_type=result["memory_type"],
            importance=int(result.get("importance", 3)),
            tags=result.get("tags", []),
        )
    except Exception as e:
        return {
            "messages": [{
                "role": "system",
                "content": f"[reflection write failed] {e}",
            }]
        }

    # reflection 자체는 사용자에게 직접 출력할 필요 없음
    return {}

