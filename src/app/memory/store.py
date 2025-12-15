from __future__ import annotations

# src/app/memory/store.py
import json
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

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


def _base_dir_from_settings() -> Path:
    # settings에 BASE_DIR이 없을 수 있어 rag_db_dir로 BASE 추정
    try:
        rag_db_dir: Path = settings.rag_db_dir  # type: ignore
        # <BASE>/data/chroma_rag 라고 가정하면 parents[1] = <BASE>
        return rag_db_dir.parents[1]
    except Exception:
        return Path.cwd()


def get_memory_db_dir() -> Path:
    # 1) settings에 memory_db_dir가 있으면 사용
    if hasattr(settings, "memory_db_dir"):
        return Path(getattr(settings, "memory_db_dir"))
    # 2) 없으면 <BASE>/data/chroma_memory 사용
    base = _base_dir_from_settings()
    return base / "data" / "chroma_memory"


def get_memory_collection_name() -> str:
    if hasattr(settings, "memory_collection_name"):
        return str(getattr(settings, "memory_collection_name"))
    return "course_memory"


def get_mem_client() -> PersistentClient:
    global _mem_client
    if _mem_client is None:
        db_dir = get_memory_db_dir()
        db_dir.mkdir(parents=True, exist_ok=True)
        _mem_client = chromadb.PersistentClient(path=str(db_dir))
    return _mem_client


def get_mem_collection():
    client = get_mem_client()
    return client.get_or_create_collection(
        name=get_memory_collection_name(),
        metadata={"hnsw:space": "cosine"},
    )


def get_mem_embedder() -> SentenceTransformer:
    """
    ⚠️ 중요:
    - SentenceTransformer는 meta tensor 상태에서 .to(device)를 호출하면 터질 수 있음
    - 따라서 최초 생성 시 device를 'cpu'로 명시 고정
    """
    global _mem_embedder
    if _mem_embedder is None:
        _mem_embedder = SentenceTransformer(
            settings.rag_embedding_model_name,
            device="cpu",   # ✅ meta tensor 문제 해결 핵심
        )
    return _mem_embedder


def write_memory(
    content: str,
    memory_type: MemoryType,
    importance: int = 3,
    tags: Optional[List[str]] = None,
) -> str:
    # tags 타입 방어
    if isinstance(tags, str):
        tags = [tags]

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
            # Chroma는 list metadata 불가 → JSON string으로 저장
            "tags": json.dumps(tags, ensure_ascii=False),
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
        raw_tags = meta.get("tags", "[]") if isinstance(meta, dict) else "[]"
        try:
            parsed_tags = json.loads(raw_tags) if isinstance(raw_tags, str) else raw_tags
        except Exception:
            parsed_tags = []

        out.append(
            MemoryItem(
                id=str(mem_id),
                content=str(doc),
                memory_type=str(meta.get("memory_type", "episodic")),  # type: ignore
                importance=int(meta.get("importance", 3)),
                tags=list(parsed_tags or []),
                created_at=str(meta.get("created_at", "")),
            )
        )
    return out
