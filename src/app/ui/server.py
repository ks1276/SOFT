# src/app/ui/server.py
from __future__ import annotations

from fastapi import FastAPI
import gradio as gr

from src.app.ui.gradio_app import build_gradio

# =========================
# FastAPI app
# =========================
app = FastAPI()


# =========================
# 🔥 WARM-UP (중요)
# =========================
@app.on_event("startup")
def warmup():
    """
    서버 시작 시:
    - SentenceTransformer 모델
    - Chroma DB 컬렉션
    을 미리 로드해서
    첫 질문이 느려지는 문제를 제거한다.
    """
    print("[WARMUP] start")

    # -------------------------
    # Memory
    # -------------------------
    from src.app.memory.store import (
        get_mem_collection,
        get_mem_embedder,
    )

    # -------------------------
    # RAG
    # -------------------------
    from src.app.rag.pipeline import (
        get_rag_collection,
        get_rag_embedder,
    )

    # 1️⃣ Chroma 컬렉션 미리 열기 (disk I/O warm)
    get_mem_collection()
    get_rag_collection()

    # 2️⃣ 임베딩 모델 로드
    mem_embedder = get_mem_embedder()
    rag_embedder = get_rag_embedder()

    # 3️⃣ 실제 forward 1회 (lazy init 제거)
    mem_embedder.encode(
        ["warmup"],
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    rag_embedder.encode(
        ["warmup"],
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    print("[WARMUP] done")


# =========================
# Gradio UI mount
# =========================
demo = build_gradio()
app = gr.mount_gradio_app(app, demo, path="/ui")


# =========================
# Root endpoint
# =========================
@app.get("/")
def root():
    return {"ok": True, "ui": "/ui"}
