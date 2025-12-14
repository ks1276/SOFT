# src/app/ui/gradio_app.py
from __future__ import annotations

import uuid
import gradio as gr

from src.app.graph.app import build_app

# 프로세스 시작 시 1번만 생성
APP = build_app(enable_interrupt=False)

TEST_1 = "123*987 계산해줘"
TEST_2 = "PDF에서 방금 넣은 문서 내용 요약해줘"
TEST_3 = "사용자는 Gradio UI를 원함을 profile로 importance 4 tags ui로 저장해줘"


def _append(chat_history: list[dict], role: str, content: str) -> list[dict]:
    chat_history = chat_history or []
    chat_history.append({"role": role, "content": content})
    return chat_history


def _invoke(user_text: str, chat_history: list[dict], thread_id: str):
    user_text = (user_text or "").strip()
    if not user_text:
        return chat_history, ""  # no-op

    cfg = {"configurable": {"thread_id": thread_id}}

    # LangGraph state 입력(최소)
    st = {
        "messages": [{"role": "user", "content": user_text}],
        "tool_calls": None,
        "steps": 0,
    }

    out = APP.invoke(st, config=cfg)
    msg = out["messages"][-1]
    assistant_text = msg.content if hasattr(msg, "content") else str(msg)

    chat_history = _append(chat_history, "user", user_text)
    chat_history = _append(chat_history, "assistant", assistant_text)
    return chat_history, ""


def _run_test(prompt: str, chat_history: list[dict], thread_id: str):
    return _invoke(prompt, chat_history, thread_id)


def build_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("## SOFT Agent (LangGraph + Tools/RAG/Memory)")

        thread = gr.State(str(uuid.uuid4()))
        chat = gr.Chatbot(height=420, type="messages")

        with gr.Row():
            inp = gr.Textbox(label="메시지", placeholder="질문을 입력하세요…", scale=8)
            btn = gr.Button("Send", scale=2)

        with gr.Row():
            t1 = gr.Button("TEST 1: 계산기")
            t2 = gr.Button("TEST 2: RAG 요약")
            t3 = gr.Button("TEST 3: 메모리 저장")

        with gr.Row():
            reset = gr.Button("New Thread")

        # 전송
        btn.click(_invoke, inputs=[inp, chat, thread], outputs=[chat, inp])
        inp.submit(_invoke, inputs=[inp, chat, thread], outputs=[chat, inp])

        # 테스트 버튼들
        t1.click(lambda h, tid: _run_test(TEST_1, h, tid), inputs=[chat, thread], outputs=[chat, inp])
        t2.click(lambda h, tid: _run_test(TEST_2, h, tid), inputs=[chat, thread], outputs=[chat, inp])
        t3.click(lambda h, tid: _run_test(TEST_3, h, tid), inputs=[chat, thread], outputs=[chat, inp])

        # 새 대화(새 thread_id + chat 초기화)
        def new_thread():
            return [], str(uuid.uuid4())

        reset.click(new_thread, outputs=[chat, thread])

        gr.Markdown(
            "- TEST 2가 동작하려면 RAG 인덱싱이 되어 있어야 합니다.\n"
            "- thread_id를 유지해서 같은 대화를 이어갑니다."
        )

    return demo
