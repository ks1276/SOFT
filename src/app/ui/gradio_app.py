# src/app/ui/gradio_app.py
from __future__ import annotations

import uuid
import gradio as gr

from src.app.graph.app import build_app

# 앱은 프로세스 시작 시 1번만 생성(속도/메모리 안정)
APP = build_app(enable_interrupt=False)

def _invoke(user_text: str, chat_history: list[tuple[str, str]], thread_id: str):
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

    chat_history = chat_history + [(user_text, assistant_text)]
    return chat_history, ""  # chat 업데이트, 입력창 비우기

def build_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("## SOFT Agent (LangGraph + Tools/RAG/Memory)")
        thread = gr.State(str(uuid.uuid4()))
        chat = gr.Chatbot(height=420)
        inp = gr.Textbox(label="메시지", placeholder="질문을 입력하세요…")
        btn = gr.Button("Send")

        btn.click(_invoke, inputs=[inp, chat, thread], outputs=[chat, inp])
        inp.submit(_invoke, inputs=[inp, chat, thread], outputs=[chat, inp])

        # thread_id 새로 만들기(새 대화)
        def new_thread():
            return [], str(uuid.uuid4())

        reset = gr.Button("New Thread")
        reset.click(new_thread, outputs=[chat, thread])

        gr.Markdown("※ thread_id를 유지해서 같은 대화를 이어갑니다.")
    return demo
