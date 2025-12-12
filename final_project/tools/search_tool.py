# tools/search_tool.py
import os
from typing import Dict, Any
import requests
from pydantic import BaseModel, Field, conint

from .tool_spec import ToolSpec


class WebSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="검색어. 예: '서울 날씨', 'Python ast eval security'",
    )
    num_results: conint(ge=1, le=5) = Field( # type: ignore
        3,
        description="가져올 검색 결과 개수 (1~5)",
    )


def google_web_search(input: WebSearchInput) -> Dict[str, Any]:
    api_key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("GOOGLE_CSE_ID")

    if not api_key or not cx:
        # 환경변수 설정 안 되어 있으면 에러만 리턴
        return {
            "ok": False,
            "error": "GOOGLE_API_KEY 또는 GOOGLE_CSE_ID 환경 변수가 설정되지 않았습니다.",
            "results": [],
        }

    params = {
        "key": api_key,
        "cx": cx,
        "q": input.query,
        "num": input.num_results,
    }

    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])

        results = []
        for item in items[: input.num_results]:
            results.append(
                {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                }
            )

        return {
            "ok": True,
            "query": input.query,
            "results": results,
        }

    except Exception as e:
        return {
            "ok": False,
            "query": input.query,
            "error": str(e),
            "results": [],
        }


def get_search_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="web_search",
        description="Google Custom Search API를 사용하여 웹을 검색합니다.",
        input_model=WebSearchInput,
        handler=google_web_search,
    )
