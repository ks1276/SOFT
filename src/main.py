# src/main.py
from __future__ import annotations

from src.app.agent.simple_tool_agent import run_once


def main() -> None:
    print("=== 간단 Function Calling 에이전트 테스트 ===")

    queries = [
        "지금 시간이 어떻게 돼?",
        "1 + 2 * 3 계산해줘.",
        "검색 툴을 사용해서 예시 결과를 보여줘.",
    ]

    for q in queries:
        print(f"\n[사용자] {q}")
        answer = run_once(q)
        print(f"[에이전트] {answer}")


if __name__ == "__main__":
    main()
