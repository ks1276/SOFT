# src/app/tools/basic.py
from __future__ import annotations

import datetime
import math
from typing import Optional

from pydantic import BaseModel, Field

from src.app.tools.__base__ import tool


# ---------- 1. 검색 툴 (더미 구현) ----------

class SearchInput(BaseModel):
    query: str = Field(..., description="사용자가 찾고 싶은 검색어")
    top_k: int = Field(3, description="최대 몇 개의 결과를 반환할지")


@tool(
    name="search",
    description="간단한 더미 검색을 수행합니다. (지금은 실제 웹 검색이 아니라 예시 텍스트를 반환합니다.)",
    input_model=SearchInput,
)
def search_tool(args: SearchInput) -> str:
    """
    간단한 더미 검색 함수.
    나중에 실제 웹 검색 API (예: SerpAPI, custom backend 등)로 교체하면 됨.
    """
    # TODO: 나중에 실제 검색 API 연동
    return f"[검색 결과 더미] '{args.query}' 에 대한 상위 {args.top_k}개 결과를 (추후 구현) 여기서 반환합니다."


# ---------- 2. 계산 툴 ----------

class CalculatorInput(BaseModel):
    expression: str = Field(
        ...,
        description=(
            "계산할 수식. +, -, *, /, 괄호, pow, sqrt 등 기본적인 연산만 사용하세요. "
            "예: '1 + 2 * 3', 'sqrt(2) ** 2'"
        ),
    )


@tool(
    name="calculator",
    description="간단한 수식을 계산합니다. 복잡한 수식은 여러 단계로 나누어서 호출하세요.",
    input_model=CalculatorInput,
)
def calculator_tool(args: CalculatorInput) -> str:
    """
    안전한 범위 내에서 수식을 계산하는 툴.
    - eval 을 그대로 쓰지 않고, 허용된 이름만 노출된 작은 네임스페이스를 사용한다.
    """
    allowed_names = {
        "sqrt": math.sqrt,
        "pow": pow,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "pi": math.pi,
        "e": math.e,
    }

    expression = args.expression

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
    except Exception as e:
        return f"수식 계산 중 오류가 발생했습니다: {e!r}"

    return f"{expression} = {result}"


# ---------- 3. 현재 시간 툴 ----------

class TimeInput(BaseModel):
    timezone: Optional[str] = Field(
        None,
        description="원하는 시간대. 지정하지 않으면 서버의 로컬 시간을 사용합니다. 예: 'Asia/Seoul'",
    )


@tool(
    name="get_time",
    description="현재 날짜와 시간을 문자열로 알려줍니다.",
    input_model=TimeInput,
)
def time_tool(args: TimeInput) -> str:
    """
    간단한 시간 툴. 지금은 timezone 을 실제로 적용하지 않고,
    서버 로컬 시간을 ISO 형식으로 반환한다.
    (나중에 pytz/zoneinfo 로 제대로 처리하도록 개선 가능)
    """
    now = datetime.datetime.now()
    iso = now.isoformat(timespec="seconds")
    if args.timezone:
        return f"현재 ({args.timezone} 기준이 아닐 수 있음) 서버 시간: {iso}"
    return f"현재 서버 시간: {iso}"
