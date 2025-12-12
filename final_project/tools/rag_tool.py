# tools/rag_tool.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import os

from pydantic import BaseModel, Field, conint

from .tool_spec import ToolSpec
from rag.vectordb import RAGVectorStore


# --- 입력 스키마 ---
class RAGSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="검색할 자연어 질의. 예: 'LangGraph에서 interrupt_before가 뭐야?'",
    )
    top_k: conint(ge=1, le=10) = Field( # type: ignore
        5,
        description="반환할 문서 청크 개수 (1~10)",
    )


# --- Lazy Singleton VectorStore (환경변수로 경로 설정 가능) ---
_vector_store: Optional[RAGVectorStore] = None


def get_vector_store() -> RAGVectorStore:
    global _vector_store
    if _vector_store is None:
        db_dir = os.getenv("RAG_DB_DIR", "./chroma_db")
        collection_name = os.getenv("RAG_COLLECTION", "project_docs")
        _vector_store = RAGVectorStore(db_dir=db_dir, collection_name=collection_name)
    return _vector_store


# --- Tool Handler ---
def rag_search_handler(input: RAGSearchInput) -> Dict[str, Any]:
    vs = get_vector_store()
    try:
        results = vs.query(
            query_text=input.query,
            top_k=input.top_k,
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


def get_rag_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="rag_search",
        description="RAG Vector DB(Chroma)에 색인된 프로젝트 PDF에서 관련 내용을 검색합니다.",
        input_model=RAGSearchInput,
        handler=rag_search_handler,
    )
