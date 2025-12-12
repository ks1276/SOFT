# tools/tool_spec.py
from typing import Any, Callable, Type
from pydantic import BaseModel
from pydantic import Field

class ToolSpec(BaseModel):
    """
    하나의 Tool에 대한 메타데이터:
    - name: LLM이 호출할 함수 이름
    - description: LLM에게 보여줄 설명
    - input_model: Pydantic 입력 모델 (→ JSON Schema로 변환)
    - handler: 실제 파이썬 함수 (input_model 인스턴스를 받아 dict 반환)
    """
    name: str
    description: str
    input_model: Type[BaseModel]
    handler: Callable[[Any], dict]

    def as_openai_tool(self) -> dict:
        """
        OpenAI chat.completions.create(..., tools=[...]) 에 넣을 포맷으로 변환
        (type=function, parameters = JSON Schema)
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
