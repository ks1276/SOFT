# src/app/ui/gradio_app.py
from __future__ import annotations

import uuid
from typing import Generator, Tuple, List, Dict, Any

import gradio as gr
import openai

from src.app.graph.app import build_app
from src.app.graph.interrupt import GraphInterrupted, request_interrupt

# -------------------------
# LangGraph App (1회 생성)
# -------------------------
# ✅ Interrupt + Stream을 위해 enable_interrupt=True
APP = build_app(enable_interrupt=True)

TEST_1 = "123*987 계산해줘"
TEST_2 = "PDF에서 방금 넣은 문서 내용 요약해줘"
TEST_3 = "사용자는 Gradio UI를 원함을 profile로 importance 4 tags ui로 저장해줘"


def _append(chat_history: List[Dict[str, str]], role: str, content: str) -> List[Dict[str, str]]:
    chat_history = chat_history or []
    chat_history.append({"role": role, "content": content})
    return chat_history


def _interrupt(thread_id: str) -> None:
    """⛔ Stop 버튼: 해당 thread_id 실행을 중단하도록 state에 플래그를 남김"""
    cfg = {"configurable": {"thread_id": thread_id}}
    APP.update_state(cfg, request_interrupt)


def _invoke(user_text: str, chat_history: list, thread_id: str):
    user_text = (user_text or "").strip()
    if not user_text:
        return chat_history, ""

    cfg = {"configurable": {"thread_id": thread_id}}

    state = {
        "messages": [{"role": "user", "content": user_text}],
        "tool_calls": None,
        "steps": 0,
    }

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_text})

    try:
        # ✅ 중요: END state만 받는다 (LangGraph가 tool 루프 보장)
        out = APP.invoke(state, config=cfg)

        messages = out.get("messages", [])
        last = messages[-1]

        # ✅ AIMessage / dict 둘 다 안전 처리
        if hasattr(last, "content"):
            content = last.content
        elif isinstance(last, dict):
            content = last.get("content", "")
        else:
            content = str(last)

        chat_history.append({"role": "assistant", "content": content})
        return chat_history, ""

    except GraphInterrupted:
        chat_history.append(
            {"role": "assistant", "content": "⛔ 실행이 중단되었습니다."}
        )
        return chat_history, ""

    except Exception as e:
        chat_history.append(
            {"role": "assistant", "content": f"⚠️ 오류: {e}"}
        )
        return chat_history, ""



def _run_test(prompt: str, chat_history: List[Dict[str, str]], thread_id: str):
    return _invoke(prompt, chat_history, thread_id)


def build_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("## SOFT Agent (LangGraph + Tools / RAG / Memory / Interrupt / Stream)")

        thread = gr.State(str(uuid.uuid4()))
        chat = gr.Chatbot(height=420)

        with gr.Row():
            inp = gr.Textbox(label="메시지", placeholder="질문을 입력하세요…", scale=8)
            btn = gr.Button("Send", scale=2)
            stop = gr.Button("⛔ Stop", scale=2)

        with gr.Row():
            t1 = gr.Button("TEST 1: 계산기")
            t2 = gr.Button("TEST 2: RAG 요약")
            t3 = gr.Button("TEST 3: 메모리 저장")

        with gr.Row():
            reset = gr.Button("New Thread")

        btn.click(_invoke, inputs=[inp, chat, thread], outputs=[chat, inp])
        inp.submit(_invoke, inputs=[inp, chat, thread], outputs=[chat, inp])

        stop.click(_interrupt, inputs=[thread], outputs=[])

        # 테스트 버튼도 stream으로 실행
        t1.click(lambda h, tid: _run_test(TEST_1, h, tid), inputs=[chat, thread], outputs=[chat, inp])
        t2.click(lambda h, tid: _run_test(TEST_2, h, tid), inputs=[chat, thread], outputs=[chat, inp])
        t3.click(lambda h, tid: _run_test(TEST_3, h, tid), inputs=[chat, thread], outputs=[chat, inp])

        def new_thread():
            return [], str(uuid.uuid4())

        reset.click(new_thread, outputs=[chat, thread])

        gr.Markdown(
            """\
- TEST 2는 RAG 인덱싱이 필요합니다.
- thread_id를 유지하여 같은 대화를 이어갑니다.
- Stop 버튼은 현재 thread의 실행을 중단합니다.
"""
        )

    return demo
