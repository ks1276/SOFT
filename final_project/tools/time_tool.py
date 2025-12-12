# tools/time_tool.py
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from .tool_spec import ToolSpec


class TimeInput(BaseModel):
    city: Optional[str] = Field(
        None,
        description="도시 이름 (선택). 예: 'Seoul'",
    )


def get_time_now(input: TimeInput) -> Dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "ok": True,
        "city": input.city,
        "now": now,
        "note": "서버 로컬 시간 기준입니다.",
    }


def get_time_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="get_time",
        description="현재 시간을 ISO 형식 문자열로 반환하는 도구",
        input_model=TimeInput,
        handler=get_time_now,
    )
