# src/app/ui/gradio_app.py
from __future__ import annotations

import uuid
from typing import Any, Dict, List

import gradio as gr

from src.app.graph.app import build_app

APP = build_app(enable_interrupt=False)

TEST_1 = "123*987 계산해줘"
TEST_2 = "PDF에서 방금 넣은 문서 내용 요약해줘"
TEST_3 = "사용자는 Gradio UI를 원함을 profile로 importance 4 tags ui로 저장해줘"

# ✅ Gradio Chatbot이 요구하는 messages 포맷
# history: [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}, ...]
ChatHistory = List[Dict[str, str]]


def _append(history: ChatHistory, role: str, content: str) -> ChatHistory:
    history = history or []
    history.append({"role": role, "content": content})
    return history


def _format_event_updates(ev: Any) -> str:
    if not isinstance(ev, dict):
        return str(ev)

    parts: List[str] = []
    for node_name, upd in ev.items():
        if isinstance(upd, dict):
            keys = ", ".join(upd.keys())
            parts.append(f"[{node_name}] updated: {keys}")
        else:
            parts.append(f"[{node_name}] {upd}")
    return " | ".join(parts)


def _invoke_once(user_text: str, thread_id: str) -> str:
    cfg = {"configurable": {"thread_id": thread_id}}
    state = {
        "messages": [{"role": "user", "content": user_text}],
        "tool_calls": None,
        "steps": 0,
    }
    out = APP.invoke(state, config=cfg)
    m = out["messages"][-1]
    return m.content if hasattr(m, "content") else str(m)


def _chat_send(
    user_text: str,
    history: ChatHistory,
    thread_id: str,
    use_stream: bool,
    trace: str,
):
    """
    ✅ 항상 generator(=항상 yield)
    outputs = [chat, inp, trace_box] 이므로 항상 3개를 yield 해야 함.
    """
    user_text = (user_text or "").strip()
    history = history or []
    trace = trace or ""

    if not user_text:
        yield history, "", trace
        return

    # UI history에 user/assistant 슬롯을 먼저 만들어 둠 (갱신 보장)
    history = _append(history, "user", user_text)
    history = _append(history, "assistant", "…(처리 중)")
    yield history, "", trace

    if not use_stream:
        try:
            assistant_text = _invoke_once(user_text, thread_id)
            history[-1] = {"role": "assistant", "content": assistant_text}
            yield history, "", trace
            return
        except Exception as e:
            history[-1] = {"role": "assistant", "content": f"⚠️ 오류: {type(e).__name__}: {e}"}
            yield history, "", trace
            return

    # stream ON: LangGraph 노드 단위 진행 로그를 trace_box에 갱신
    try:
        cfg = {"configurable": {"thread_id": thread_id}}
        state = {
            "messages": [{"role": "user", "content": user_text}],
            "tool_calls": None,
            "steps": 0,
        }

        local_lines: List[str] = []
        for ev in APP.stream(state, config=cfg, stream_mode="updates"):
            local_lines.append(_format_event_updates(ev))
            new_trace = (trace + "\n" + "\n".join(local_lines)).strip()
            yield history, "", new_trace

        # 최종 답변은 invoke 결과로 확정(토큰 스트리밍이 아니라 "노드 스트리밍"이므로)
        assistant_text = _invoke_once(user_text, thread_id)
        history[-1] = {"role": "assistant", "content": assistant_text}
        final_trace = (trace + "\n" + "\n".join(local_lines)).strip()
        yield history, "", final_trace

    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"⚠️ 오류: {type(e).__name__}: {e}"}
        yield history, "", trace


# ✅ TEST 버튼도 반드시 generator 함수로 직접 연결 (lambda 금지)
def _test1(history: ChatHistory, thread_id: str, use_stream: bool, trace: str):
    yield from _chat_send(TEST_1, history, thread_id, use_stream, trace)


def _test2(history: ChatHistory, thread_id: str, use_stream: bool, trace: str):
    yield from _chat_send(TEST_2, history, thread_id, use_stream, trace)


def _test3(history: ChatHistory, thread_id: str, use_stream: bool, trace: str):
    yield from _chat_send(TEST_3, history, thread_id, use_stream, trace)


def build_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("## SOFT Agent (LangGraph + Tools / RAG / Memory)")

        thread = gr.State(str(uuid.uuid4()))

        chat = gr.Chatbot(height=420)  # ✅ messages 포맷 반환으로 맞춤
        use_stream = gr.Checkbox(value=False, label="LangGraph stream(노드 단위 진행 로그 보기)")
        trace_box = gr.Textbox(
            label="Stream Trace (노드별 업데이트)",
            value="",
            lines=10,
            interactive=False,
        )

        with gr.Row():
            inp = gr.Textbox(label="메시지", placeholder="질문을 입력하세요…", scale=8)
            btn = gr.Button("Send", scale=2)

        with gr.Row():
            t1 = gr.Button("TEST 1: 계산기")
            t2 = gr.Button("TEST 2: RAG 요약")
            t3 = gr.Button("TEST 3: 메모리 저장")

        with gr.Row():
            reset = gr.Button("New Thread")
            clear_trace = gr.Button("Clear Trace")

        # Send / Enter
        btn.click(
            _chat_send,
            inputs=[inp, chat, thread, use_stream, trace_box],
            outputs=[chat, inp, trace_box],
        )
        inp.submit(
            _chat_send,
            inputs=[inp, chat, thread, use_stream, trace_box],
            outputs=[chat, inp, trace_box],
        )

        # Tests (lambda 금지)
        t1.click(_test1, inputs=[chat, thread, use_stream, trace_box], outputs=[chat, inp, trace_box])
        t2.click(_test2, inputs=[chat, thread, use_stream, trace_box], outputs=[chat, inp, trace_box])
        t3.click(_test3, inputs=[chat, thread, use_stream, trace_box], outputs=[chat, inp, trace_box])

        # New Thread: history/trace는 비우고 thread만 새로
        def new_thread():
            return [], str(uuid.uuid4()), ""

        reset.click(new_thread, outputs=[chat, thread, trace_box])

        clear_trace.click(lambda: "", outputs=[trace_box])

        gr.Markdown(
            "- TEST 2는 RAG 인덱싱이 되어 있어야 결과가 나옵니다.\n"
            "- stream 체크 시, LangGraph stream_mode='updates' 기반 노드 진행 로그를 trace에 출력합니다."
        )

    return demo
