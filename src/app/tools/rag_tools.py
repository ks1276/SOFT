# src/app/tools/rag_tools.py
from __future__ import annotations

from pydantic import BaseModel, Field

from src.app.rag.pipeline import query_rag, format_rag_answer
from src.app.tools.__base__ import tool


class RagQueryInput(BaseModel):
    query: str = Field(..., description="검색할 질문/키워드")
    top_k: int = Field(5, ge=1, le=10, description="가져올 청크 개수")


@tool(
    name="rag_search",
    description="PDF 문서에서 관련 내용을 벡터 검색으로 찾아 요약용 근거를 반환합니다.",
    input_model=RagQueryInput,
)
def rag_search_tool(args: RagQueryInput) -> str:
    res = query_rag(args.query, args.top_k)
    return format_rag_answer(res)
