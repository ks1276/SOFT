# src/app/llm/client.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.app.config.settings import settings  # ← 새로 추가

# ---- 기본 설정 ----

DEFAULT_MODEL: str = settings.openai_model
DEFAULT_TEMPERATURE: float = settings.openai_temperature

SYSTEM_PROMPT: str = """\
You are a helpful AI assistant that can use tools with a ReAct-style loop.
- 먼저 사용자의 질문을 이해하고, 필요하면 내부적으로 조용히 생각합니다.
- 계산/검색/문서 조회/메모리 조회 등 도구가 필요하면 제공된 tools를 사용합니다.
- tool 결과를 보고 다시 생각한 뒤, 최종 답변은 한국어로 명확하게 정리해서 제공합니다.
- 도구가 필요한 작업(예: 최신 정보 검색, 복잡한 계산 등)은 반드시 tool_calls로 처리하고,
  추측으로 지어내지 않습니다.
"""

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """
    OpenAI 클라이언트를 전역에서 하나만 생성해서 재사용한다.
    API 키는 .env 에서 읽은 settings.openai_api_key 를 사용.
    """
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def chat_raw(
    messages: List[Dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = "auto",
    max_tokens: Optional[int] = None,
) -> Any:
    client = get_client()

    params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if tools:
        params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    response = client.chat.completions.create(**params)
    return response


def chat_simple(
    user_content: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
) -> str:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    resp = chat_raw(
        messages,
        model=model,
        temperature=temperature,
        tools=None,
        tool_choice="none",
        max_tokens=max_tokens,
    )

    msg = resp.choices[0].message
    return msg.content or ""


if __name__ == "__main__":
    print(">>> LLM 래퍼 테스트: '안녕, 자기소개해줘'")
    answer = chat_simple("안녕, 자기소개해줘")
    print(answer)
