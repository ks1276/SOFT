from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
import gradio as gr
import uuid

from main_agent import graph

app = FastAPI(title="LangGraph Agent API")


# =========================
# Pydantic Models
# =========================

class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    output: str


# =========================
# LangGraph helper
# =========================

def extract_messages_from_event(node_state):
    if isinstance(node_state, dict):
        return node_state.get("messages", [])
    elif isinstance(node_state, list):
        return node_state
    return []


def get_assistant_text(msgs):
    """마지막 assistant 메시지의 content만 추출"""
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content", "")
    return ""


# =========================
# Gradio Chat
# =========================

def gradio_chat(message, history, state):
    if state is None:
        state = {"thread_id": str(uuid.uuid4())}

    if history is None:
        history = []

    config = RunnableConfig(
        configurable={"thread_id": state["thread_id"]}
    )

    # ✅ LangGraph에는 dict 메시지
    user_input = {
        "messages": [
            {"role": "user", "content": message}
        ]
    }

    final_text = ""

    for event in graph.stream(
        input=user_input,
        config=config,
        stream_mode="values",
    ):
        for _, node_state in event.items():
            msgs = extract_messages_from_event(node_state)
            if msgs:
                final_text = get_assistant_text(msgs)

    # ✅ Gradio Chatbot은 (user, assistant) 튜플만
    history.append((message, final_text))
    return history, state


# =========================
# Gradio UI
# =========================

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 LangGraph AI Agent")

    chatbot = gr.Chatbot()   # ✅ type 절대 쓰지 말 것 (Gradio 6.1)
    msg = gr.Textbox(label="메시지 입력")
    state = gr.State()

    msg.submit(
        gradio_chat,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, state],
    )


# =========================
# Mount
# =========================

app = gr.mount_gradio_app(app, demo, path="/")
