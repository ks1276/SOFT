# src/app/ui/server.py
from __future__ import annotations

from fastapi import FastAPI
import gradio as gr

from src.app.ui.gradio_app import build_gradio

app = FastAPI()

demo = build_gradio()
app = gr.mount_gradio_app(app, demo, path="/ui")

@app.get("/")
def root():
    return {"ok": True, "ui": "/ui"}
