# src/app/memory/reflection.py
from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

MEMORY_EXTRACTOR_PROMPT = """You are a memory extraction assistant.
Your task:
- Read the given conversation between a user and an assistant.
- Decide whether there is any information that should be stored as long-term memory.
- Long-term memories include:
- User's stable preferences.
- Long-term projects or goals.
- Important facts that will likely be useful in future conversations.
- Do NOT store short-lived/trivial facts or sensitive personal data unless explicitly requested.
Memory types:
- "profile": User’s preferences, long-term goals.
- "episodic": Summary of this session/turn (what was done/decided).
- "knowledge": Reusable general facts/explanations.
Output:
Return ONE JSON object:
{"should_write_memory": false}
or if true:
{
  "should_write_memory": true,
  "memory_type": "profile" | "episodic" | "knowledge",
  "importance": 1~5,
  "content": "...",
  "tags": ["..."]
}
"""

def build_snippet(history: List[Any], user_message: str, final_answer: str) -> str:
    # history가 messages라면 마지막 몇 개만 요약/발췌
    # (너무 길면 extractor가 흔들립니다)
    recent = history[-8:] if len(history) > 8 else history
    recent_text = "\n".join([str(m) for m in recent])
    return f"[RECENT]\n{recent_text}\n\n[USER]\n{user_message}\n\n[ASSISTANT_FINAL]\n{final_answer}"

def run_memory_extractor(llm: ChatOpenAI, snippet: str) -> Dict[str, Any]:
    resp = llm.invoke([SystemMessage(content=MEMORY_EXTRACTOR_PROMPT), HumanMessage(content=snippet)])
    text = resp.content.strip()
    try:
        return json.loads(text)
    except Exception:
        # 모델이 JSON을 약간 망가뜨리면 최소 방어
        return {"should_write_memory": False}
