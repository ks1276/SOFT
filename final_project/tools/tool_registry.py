# tools/tool_registry.py
from typing import Dict, Any, List
from pydantic import BaseModel

from .tool_spec import ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register_tool(self, spec: ToolSpec) -> None:
        """ToolSpec을 이름으로 등록"""
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' is already registered.")
        self._tools[spec.name] = spec

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM이 tool_call로 넘긴 name, arguments(dict)를 받아
        - Pydantic으로 검증
        - 실제 handler 실행
        - dict 결과 반환
        """
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")

        spec = self._tools[name]

        # Pydantic 입력 모델로 검증
        if not issubclass(spec.input_model, BaseModel):
            raise TypeError(f"Tool '{name}' has invalid input_model type.")

        input_obj = spec.input_model(**args)
        return spec.handler(input_obj)

    def list_openai_tools(self) -> List[dict]:
        """OpenAI tools=[] 파라미터에 넣을 수 있는 리스트 반환"""
        return [spec.as_openai_tool() for spec in self._tools.values()]
