# src/app/tools/__base__.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field


@dataclass
class ToolSpec:
    """
    하나의 Function Calling tool 을 표현하는 스펙.

    - name: OpenAI tool 에서 사용할 function 이름
    - description: tool 의 용도 설명 (한국어/영어 아무거나, LLM이 이해 가능하면 OK)
    - input_model: Pydantic BaseModel (arguments 스키마)
    - func: 실제 파이썬 함수. 인자로 input_model 인스턴스를 받고, 결과를 반환.
    """
    name: str
    description: str
    input_model: Type[BaseModel]
    func: Callable[[BaseModel], Any]

    def to_openai_tool(self) -> Dict[str, Any]:
        """
        OpenAI tools 포맷으로 변환.

        OpenAI Chat Completions v1 기준:
        tools = [
          {
            "type": "function",
            "function": {
              "name": "...",
              "description": "...",
              "parameters": { ... JSON Schema ... }
            }
          },
          ...
        ]
        """
        schema = self.input_model.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    def invoke_from_json(self, arguments: str | Dict[str, Any]) -> Any:
        """
        OpenAI tool_call 로부터 받은 arguments 를 이용해 실제 함수 실행.

        - arguments 가 str 이면 JSON string 으로 보고 dict 로 parse
        - arguments 가 dict 면 그대로 사용
        """
        if isinstance(arguments, str):
            data = json.loads(arguments or "{}")
        else:
            data = arguments

        model_instance = self.input_model(**data)
        return self.func(model_instance)


class ToolRegistry:
    """
    프로젝트 전역에서 사용할 tool 들을 등록/조회하는 레지스트리.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' is already registered.")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Tool '{name}' is not registered.")

    def list_specs(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def list_openai_tools(self) -> List[Dict[str, Any]]:
        """
        OpenAI chat.completions.create 에 그대로 넘길 tools 리스트.
        """
        return [spec.to_openai_tool() for spec in self._tools.values()]

    def invoke(self, name: str, arguments: str | Dict[str, Any]) -> Any:
        """
        tool 이름과 arguments(JSON string or dict)를 받아 실제 파이썬 함수를 실행.
        """
        spec = self.get(name)
        return spec.invoke_from_json(arguments)


# 전역 레지스트리 인스턴스
registry = ToolRegistry()


def tool(
    *,
    name: Optional[str] = None,
    description: str = "",
    input_model: Type[BaseModel],
) -> Callable[[Callable[[BaseModel], Any]], Callable[[BaseModel], Any]]:
    """
    데코레이터 형태로 ToolSpec 을 등록하기 위한 helper.

    사용 예:
        class SearchInput(BaseModel):
            query: str = Field(..., description="검색어")

        @tool(name="search", description="웹 검색을 수행합니다.", input_model=SearchInput)
        def search_tool(args: SearchInput) -> str:
            ...

    이렇게 하면 registry 에 ToolSpec 이 자동 등록되고,
    나중에 registry.list_openai_tools() 로 OpenAI tools 리스트를 넘길 수 있음.
    """
    def decorator(func: Callable[[BaseModel], Any]) -> Callable[[BaseModel], Any]:
        tool_name = name or func.__name__
        spec = ToolSpec(
            name=tool_name,
            description=description or (func.__doc__ or "").strip(),
            input_model=input_model,
            func=func,
        )
        registry.register(spec)
        return func

    return decorator
