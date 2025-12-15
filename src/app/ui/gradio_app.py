from __future__ import annotations

import uuid
from typing import Any, Dict, List

import gradio as gr

from src.app.graph.app import build_app

# 🔥 interrupt 사용
APP = build_app(enable_interrupt=True)

TEST_1 = "123*987 계산해줘"
TEST_2 = "PDF에서 방금 넣은 문서 내용 요약해줘"
TEST_3 = "사용자는 Gradio UI를 원함을 profile로 importance 4 tags ui로 저장해줘"

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
    messages = out.get("messages", [])

    if not messages:
        return "⚠️ 응답이 생성되지 않았습니다."

    m = messages[-1]
    return m.get("content", "") if isinstance(m, dict) else str(m)


def _chat_send(
    user_text: str,
    history: ChatHistory,
    thread_id: str,
    use_stream: bool,
    trace: str,
):
    user_text = (user_text or "").strip()
    history = history or []
    trace = trace or ""

    if not user_text:
        yield history, "", trace
        return

    history = _append(history, "user", user_text)
    history = _append(history, "assistant", "…(처리 중)")
    yield history, "", trace

    cfg = {"configurable": {"thread_id": thread_id}}
    state = {
        "messages": [{"role": "user", "content": user_text}],
        "tool_calls": None,
        "steps": 0,
    }

    # ================= STREAM OFF =================
    if not use_stream:
        try:
            APP.invoke(state, config=cfg)
            snapshot = APP.get_state(cfg)

            # 🔥 tool 직전 interrupt 감지
            if snapshot.next and "tool" in snapshot.next:
                history[-1] = {
                    "role": "assistant",
                    "content": (
                        "⛔ tool 실행 직전에서 중단되었습니다.\n\n"
                        "▶ Continue : 그대로 진행\n"
                        "✏️ Edit & Resume : 내용 수정 후 진행"
                    ),
                }
                yield history, "", trace
                return

            # interrupt 없을 때만 최종 답변
            assistant_text = _invoke_once(user_text, thread_id)
            history[-1] = {"role": "assistant", "content": assistant_text}
            yield history, "", trace
            return

        except Exception as e:
            history[-1] = {"role": "assistant", "content": f"⚠️ 오류: {e}"}
            yield history, "", trace
            return

    # ================= STREAM ON =================
    try:
        local_lines: List[str] = []

        for ev in APP.stream(state, config=cfg, stream_mode="updates"):
            local_lines.append(_format_event_updates(ev))
            new_trace = (trace + "\n" + "\n".join(local_lines)).strip()
            yield history, "", new_trace

        snapshot = APP.get_state(cfg)

        # 🔥 tool 직전 interrupt 감지
        if snapshot.next and "tool" in snapshot.next:
            history[-1] = {
                "role": "assistant",
                "content": (
                    "⛔ tool 실행 직전에서 중단되었습니다.\n\n"
                    "▶ Continue : 그대로 진행\n"
                    "✏️ Edit & Resume : 내용 수정 후 진행"
                ),
            }
            yield history, "", new_trace
            return

        # interrupt 없을 때만 최종 답변
        assistant_text = _invoke_once(user_text, thread_id)
        history[-1] = {"role": "assistant", "content": assistant_text}
        yield history, "", new_trace

    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"⚠️ 오류: {e}"}
        yield history, "", trace


# ================= Resume =================

def _resume(history: ChatHistory, thread_id: str, trace: str):
    cfg = {"configurable": {"thread_id": thread_id}}
    result = APP.invoke(None, config=cfg)

    messages = result.get("messages", [])
    if not messages:
        history = _append(history, "assistant", "⚠️ 재개할 중단 상태가 없습니다.")
        return history, "", trace

    m = messages[-1]
    content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
    history = _append(history, "assistant", content)
    return history, "", trace



def _edit_and_resume(
    history: ChatHistory,
    thread_id: str,
    new_text: str,
    trace: str,
):
    """
    Edit = '새 질문으로 이어가기'
    - 기존 interrupt state는 버린다
    - 새 질문처럼 다시 실행
    """
    new_text = (new_text or "").strip()
    if not new_text:
        return history, "", trace

    # 🔥 핵심: 새 thread_id 생성 (state 리셋)
    new_thread_id = str(uuid.uuid4())

    # UI에는 user 입력으로 추가
    history = _append(history, "user", new_text)
    history = _append(history, "assistant", "…(처리 중)")

    # 기존 _chat_send 재사용 (절대 안 깨짐)
    yield from _chat_send(
        new_text,
        history,
        new_thread_id,
        use_stream=False,
        trace=trace,
    )





# ================= UI =================
# ===== TEST 버튼 =====

def _test1(history, thread_id, use_stream, trace):
    yield from _chat_send(TEST_1, history, thread_id, use_stream, trace)

def _test2(history, thread_id, use_stream, trace):
    yield from _chat_send(TEST_2, history, thread_id, use_stream, trace)

def _test3(history, thread_id, use_stream, trace):
    yield from _chat_send(TEST_3, history, thread_id, use_stream, trace)


def build_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("## SOFT Agent (Stream + Interrupt_before)")

        thread = gr.State(str(uuid.uuid4()))
        chat = gr.Chatbot(height=420)

        use_stream = gr.Checkbox(value=False, label="LangGraph stream (노드 단위)")
        trace_box = gr.Textbox(label="Stream Trace", lines=10, interactive=False)

        with gr.Row():
            inp = gr.Textbox(scale=8, placeholder="질문 입력")
            btn = gr.Button("Send", scale=2)

        with gr.Row():
            resume = gr.Button("▶ Continue")
            edit = gr.Button("✏️ Edit & Resume")
            edit_text = gr.Textbox(placeholder="수정할 내용")

        with gr.Row():
            t1 = gr.Button("TEST 1")
            t2 = gr.Button("TEST 2")
            t3 = gr.Button("TEST 3")

        btn.click(_chat_send, inputs=[inp, chat, thread, use_stream, trace_box], outputs=[chat, inp, trace_box])
        inp.submit(_chat_send, inputs=[inp, chat, thread, use_stream, trace_box], outputs=[chat, inp, trace_box])

        resume.click(_resume, inputs=[chat, thread, trace_box], outputs=[chat, inp, trace_box])
        edit.click(_edit_and_resume, inputs=[chat, thread, edit_text, trace_box], outputs=[chat, inp, trace_box])

        t1.click(_test1, inputs=[chat, thread, use_stream, trace_box], outputs=[chat, inp, trace_box])
        t2.click(_test2, inputs=[chat, thread, use_stream, trace_box], outputs=[chat, inp, trace_box])
        t3.click(_test3, inputs=[chat, thread, use_stream, trace_box], outputs=[chat, inp, trace_box])

    return demo
