# src/app/memory/pipeline.py
from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from src.app.config.settings import settings

MemoryType = Literal["profile", "episodic", "knowledge"]

@dataclass
class MemoryItem:
    id: str
    content: str
    memory_type: MemoryType
    importance: int
    tags: List[str]
    created_at: str

_mem_client: Optional[PersistentClient] = None
_mem_embedder: Optional[SentenceTransformer] = None

def get_mem_client() -> PersistentClient:
    global _mem_client
    if _mem_client is None:
        settings.memory_db_dir.mkdir(parents=True, exist_ok=True)
        _mem_client = chromadb.PersistentClient(path=str(settings.memory_db_dir))
    return _mem_client

def get_mem_collection():
    client = get_mem_client()
    return client.get_or_create_collection(
        name=settings.memory_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

def get_mem_embedder() -> SentenceTransformer:
    global _mem_embedder
    if _mem_embedder is None:
        _mem_embedder = SentenceTransformer(settings.rag_embedding_model_name)
    return _mem_embedder

def write_memory(
    content: str,
    memory_type: MemoryType,
    importance: int = 3,
    tags: Optional[List[str]] = None,
) -> str:
    tags = tags or []
    importance = max(1, min(int(importance), 5))
    now = datetime.datetime.now().isoformat(timespec="seconds")
    mem_id = f"mem::{now}"

    col = get_mem_collection()
    emb = get_mem_embedder().encode([content], show_progress_bar=False).tolist()[0]

    col.add(
        ids=[mem_id],
        documents=[content],
        embeddings=[emb],
        metadatas=[{
            "memory_type": memory_type,
            "importance": importance,
            "tags": tags,
            "created_at": now,
        }],
    )
    return mem_id

def read_memory(query: str, top_k: int = 5) -> List[MemoryItem]:
    col = get_mem_collection()
    qemb = get_mem_embedder().encode([query], show_progress_bar=False).tolist()

    res = col.query(
        query_embeddings=qemb,
        n_results=max(1, min(int(top_k), 10)),
        include=["documents", "metadatas"],
    )

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    out: List[MemoryItem] = []
    for mem_id, doc, meta in zip(ids, docs, metas):
        out.append(
            MemoryItem(
                id=str(mem_id),
                content=str(doc),
                memory_type=str(meta.get("memory_type", "episodic")),  # fallback
                importance=int(meta.get("importance", 3)),
                tags=list(meta.get("tags", []) or []),
                created_at=str(meta.get("created_at", "")),
            )
        )
    return out
