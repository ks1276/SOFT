# src/app/tools/memory_tools.py
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from src.app.memory.store import read_memory, write_memory

from src.app.tools.__base__ import tool

MemoryType = Literal["profile", "episodic", "knowledge"]

class ReadMemoryInput(BaseModel):
    query: str = Field(..., description="장기 메모리에서 찾고 싶은 내용")
    top_k: int = Field(5, description="가져올 결과 개수 (기본 5)")

@tool(
    name="read_memory",
    description="저장된 장기 메모리에서 현재 질문과 관련된 내용을 검색합니다.",
    input_model=ReadMemoryInput,
)
def read_memory_tool(args: ReadMemoryInput) -> str:
    items = read_memory(args.query, top_k=args.top_k)
    if not items:
        return "관련 메모리가 없습니다."

    lines = ["[MEMORY RESULTS]"]
    for i, it in enumerate(items, start=1):
        tag_str = ", ".join(it.tags) if it.tags else "-"
        lines.append(
            f"{i}) type={it.memory_type} importance={it.importance} created_at={it.created_at} tags={tag_str}\n"
            f"{it.content}"
        )
    return "\n\n---\n\n".join(lines)

class WriteMemoryInput(BaseModel):
    content: str = Field(..., description="저장할 메모리 내용(짧고 재사용 가능하게)")
    memory_type: MemoryType = Field(..., description="profile | episodic | knowledge")
    importance: int = Field(3, description="1(낮음) ~ 5(높음)")
    tags: Optional[List[str]] = Field(default=None, description="태그 리스트(선택)")

@tool(
    name="write_memory",
    description="새로운 장기 메모리를 저장합니다.",
    input_model=WriteMemoryInput,
)
def write_memory_tool(args: WriteMemoryInput) -> str:
    mem_id = write_memory(
        content=args.content,
        memory_type=args.memory_type,
        importance=args.importance,
        tags=args.tags or [],
    )
    return f"saved_memory_id={mem_id}"
