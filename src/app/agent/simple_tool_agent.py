# src/app/agent/simple_tool_agent.py
from __future__ import annotations

from typing import Any, Dict, List

from src.app.llm.client import chat_raw, SYSTEM_PROMPT
from src.app.tools.__base__ import registry
from src.app.tools import basic  # noqa: F401  # import 되어야 데코레이터가 실행되어 registry에 등록됨


def _openai_message_from_choice(msg) -> Dict[str, Any]:
    """
    OpenAI ChatCompletionMessage 객체를, 다음 호출에 쓸 수 있는 dict 형태로 변환.
    (role, content, tool_calls 만 뽑아서 사용)
    """
    data: Dict[str, Any] = {
        "role": msg.role,
        "content": msg.content,
    }
    if msg.tool_calls:
        # tool_calls도 dict 리스트로 변환
        data["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in msg.tool_calls]
    return data


def run_once(user_input: str) -> str:
    """
    1턴짜리 간단 ReAct-style tool agent.

    1) system + user 메시지로 LLM 호출 (tools 포함, tool_choice="auto")
    2) tool_calls가 있으면 실제 파이썬 함수 실행 (registry.invoke)
    3) tool 결과를 tool 메시지로 붙이고, tool_choice="none" 으로 다시 LLM 호출
    4) 최종 assistant 응답 텍스트를 반환
    """
    # 1. 초기 메시지 구성
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    # 2. 첫 번째 호출: LLM이 tool을 쓸지 말지 결정
    resp1 = chat_raw(
        messages,
        tools=registry.list_openai_tools(),
        tool_choice="auto",
    )
    msg1 = resp1.choices[0].message
    messages.append(_openai_message_from_choice(msg1))

    # tool_calls 없으면 바로 답변했다고 보고 content 리턴
    if not msg1.tool_calls:
        return msg1.content or ""

    # 3. tool_calls 있으면 실제 파이썬 함수 실행
    for tc in msg1.tool_calls:
        tool_name = tc.function.name
        arguments = tc.function.arguments  # JSON string
        result = registry.invoke(tool_name, arguments)

        # tool 메시지를 messages에 추가
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tool_name,
                "content": str(result),
            }
        )

    # 4. tool 결과를 보고 최종 답변 생성 (이제는 tool_choice="none")
    resp2 = chat_raw(
        messages,
        tools=registry.list_openai_tools(),  # 넘겨도 되지만,
        tool_choice="none",                  # 더 이상 새 tool 호출은 금지
    )
    msg2 = resp2.choices[0].message
    return msg2.content or ""
