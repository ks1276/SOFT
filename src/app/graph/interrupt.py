from __future__ import annotations

from typing import Any, Dict

# =====================================
# Interrupt flag key
# =====================================
INTERRUPT_FLAG = "_interrupted"


class GraphInterrupted(Exception):
    """그래프 실행이 사용자에 의해 중단되었을 때 발생"""
    pass


def mark_interrupted(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    외부(UI Stop 버튼 등)에서 호출:
    해당 thread 실행 state에 interrupt 플래그를 남김
    """
    return {INTERRUPT_FLAG: True}


# ✅ UI에서 쓰는 이름이 request_interrupt였으므로 alias 제공
def request_interrupt(state: Dict[str, Any]) -> Dict[str, Any]:
    return mark_interrupted(state)


def raise_if_interrupted(state: Dict[str, Any]) -> None:
    """
    각 노드에서 호출하여 interrupt 상태면 즉시 예외 발생
    """
    if isinstance(state, dict) and state.get(INTERRUPT_FLAG):
        raise GraphInterrupted("Execution interrupted by user")
