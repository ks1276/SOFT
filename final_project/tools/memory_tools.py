# tools/memory_tools.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal

from pydantic import BaseModel, Field, conint

from .tool_spec import ToolSpec
from memory.memory_store import MemoryStore


# -------------------------
# Singleton MemoryStore
# -------------------------
_memory_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()  # 기본 경로 ./memory_db
    return _memory_store


# -------------------------
# Write Memory Tool
# -------------------------
class WriteMemoryInput(BaseModel):
    content: str = Field(
        ...,
        description="저장할 메모리 내용 (사용자 정보, 선호, 프로젝트 진행상황 등)",
    )
    memory_type: Literal["profile", "episodic", "knowledge"] = Field(
        "episodic",
        description="메모리 종류: profile(사용자 프로필), episodic(경험), knowledge(지식)",
    )
    importance: conint(ge=1, le=5) = Field( # type: ignore
        3,
        description="중요도 1(낮음) ~ 5(매우 높음)",
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="검색/필터 용 태그 목록",
    )


def write_memory_handler(input: WriteMemoryInput) -> Dict[str, Any]:
    store = get_memory_store()
    mem_id = store.add_memory(
        content=input.content,
        memory_type=input.memory_type,
        importance=input.importance,
        tags=input.tags,
    )
    return {
        "ok": True,
        "memory_id": mem_id,
    }


def get_write_memory_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="write_memory",
        description="사용자/대화/지식에 관한 새로운 메모리를 장기 저장소에 기록합니다.",
        input_model=WriteMemoryInput,
        handler=write_memory_handler,
    )


# -------------------------
# Read Memory Tool
# -------------------------
class ReadMemoryInput(BaseModel):
    query: str = Field(
        ...,
        description="찾고 싶은 내용에 대한 자연어 질의. 예: '이 사용자가 선호하는 언어'",
    )
    top_k: conint(ge=1, le=10) = Field( # type: ignore
        5,
        description="가져올 메모리 개수 (1~10)",
    )
    memory_type: Optional[Literal["profile", "episodic", "knowledge"]] = Field(
        None,
        description="특정 타입 메모리만 검색하고 싶을 때 지정",
    )


def read_memory_handler(input: ReadMemoryInput) -> Dict[str, Any]:
    store = get_memory_store()
    results = store.search_memories(
        query=input.query,
        top_k=input.top_k,
        memory_type=input.memory_type,
    )
    return {
        "ok": True,
        "query": input.query,
        "results": results,
    }


def get_read_memory_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="read_memory",
        description="장기 메모리에서 사용자의 과거 정보나 경험을 검색합니다.",
        input_model=ReadMemoryInput,
        handler=read_memory_handler,
    )
