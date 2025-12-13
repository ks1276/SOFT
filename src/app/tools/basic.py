# src/app/tools/basic.py
from __future__ import annotations

import datetime
import math
import os
from typing import Optional

import requests
from pydantic import BaseModel, Field

from src.app.tools.__base__ import tool


# ---------- 1. 검색 툴 (Google Custom Search JSON API) ----------

class SearchInput(BaseModel):
    query: str = Field(..., description="사용자가 찾고 싶은 검색어")
    top_k: int = Field(3, description="최대 몇 개의 결과를 반환할지 (1~10 권장)")
    site: Optional[str] = Field(None, description="특정 도메인만 검색 (예: docs.python.org)")


@tool(
    name="search",
    description="Google Programmable Search Engine(CSE) 기반 웹 검색을 수행합니다.",
    input_model=SearchInput,
)
def search_tool(args: SearchInput) -> str:
    """
    Google Custom Search JSON API를 호출해 검색 결과를 텍스트로 반환합니다.

    환경변수(.env)
    - GOOGLE_API_KEY: Google Cloud Console에서 발급한 API Key
    - GOOGLE_CSE_ID: Programmable Search Engine의 Search engine ID (cx)
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")  # cx

    if not api_key or not cse_id:
        return (
            "ERROR: GOOGLE_API_KEY 또는 GOOGLE_CSE_ID가 설정되지 않았습니다.\n"
            "(.env에 GOOGLE_API_KEY=..., GOOGLE_CSE_ID=... 추가 후 재실행하세요.)"
        )

    num = max(1, min(int(args.top_k), 10))

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": args.query,
        "num": num,
    }
    if args.site:
        params["siteSearch"] = args.site

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as e:
        # API가 에러 메시지를 JSON으로 주는 경우가 많아서 최대한 노출
        try:
            err = resp.json()
        except Exception:
            err = resp.text if "resp" in locals() else ""
        return f"ERROR: 구글 검색 API HTTP 오류: {e}\n{err}"
    except Exception as e:
        return f"ERROR: 구글 검색 호출 실패: {e!r}"

    items = data.get("items") or []
    if not items:
        return f"검색 결과 없음: {args.query}"

    # 에이전트가 쓰기 좋게: 번호 + 제목 + 링크 + 요약
    lines = [f"[Google Search] query='{args.query}' (top {num})"]
    for i, it in enumerate(items, start=1):
        title = (it.get("title") or "").strip()
        link = (it.get("link") or "").strip()
        snippet = (it.get("snippet") or "").strip()
        lines.append(f"{i}. {title}\n   - {link}\n   - {snippet}")

    return "\n".join(lines)


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
