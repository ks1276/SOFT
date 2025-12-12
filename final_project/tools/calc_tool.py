# tools/calc_tool.py
import ast
import operator
from typing import Dict, Any
from pydantic import BaseModel, Field

from .tool_spec import ToolSpec


class CalcInput(BaseModel):
    expression: str = Field(
        ...,
        description="사칙연산 수식. 예: '2+3*4-5/2'",
    )


# 허용할 연산자만 정의
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Num):  # 파이썬 <3.8
        return node.n
    if isinstance(node, ast.Constant):  # 파이썬 3.8+
        if isinstance(node.value, (int, float)):
            return node.value
        else:
            raise ValueError("숫자만 허용됩니다.")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"허용되지 않은 연산자: {op_type}")
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return _ALLOWED_OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"허용되지 않은 단항 연산자: {op_type}")
        operand = _eval_ast(node.operand)
        return _ALLOWED_OPERATORS[op_type](operand)
    raise ValueError(f"지원하지 않는 표현식 타입: {type(node)}")


def eval_expression(input: CalcInput) -> Dict[str, Any]:
    try:
        tree = ast.parse(input.expression, mode="eval")
        result = _eval_ast(tree.body)
        return {
            "ok": True,
            "expression": input.expression,
            "result": result,
        }
    except Exception as e:
        return {
            "ok": False,
            "expression": input.expression,
            "error": str(e),
        }


def get_calc_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="calculator",
        description="간단한 수식(사칙연산)을 계산하는 계산기 도구",
        input_model=CalcInput,
        handler=eval_expression,
    )
