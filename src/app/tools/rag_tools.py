# src/app/tools/rag_tools.py
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from src.app.rag.pipeline import query_rag, format_rag_answer
from src.app.tools.__base__ import tool


class RagQueryInput(BaseModel):
    query: str = Field(..., description="사용자가 PDF 문서에서 찾고 싶은 내용(질문)")
    top_k: int = Field(5, description="가져올 관련 문서 청크 개수 (기본 5)")


@tool(
    name="rag_search",
    description=(
        "PDF 문서들을 기반으로 RAG 검색을 수행합니다. "
        "질문에 관련된 문서 조각들을 찾아서 원문 텍스트를 반환합니다."
    ),
    input_model=RagQueryInput,
)
def rag_search_tool(args: RagQueryInput) -> str:
    """
    RAG 검색 tool.
    - 먼저 query_rag 로 상위 top_k 개의 청크를 찾고
    - LLM 이 직접 참조할 수 있도록 포맷팅된 문자열을 반환한다.
    """
    chunks = query_rag(args.query, top_k=args.top_k)
    return format_rag_answer(chunks)
