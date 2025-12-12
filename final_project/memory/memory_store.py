# memory/memory_store.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import uuid

import chromadb
from chromadb.config import Settings

from rag.embedder import EmbeddingModel  # RAG에서 쓰던 모델 재사용


class MemoryStore:
    """
    Long-term Memory용 Vector DB (Chroma Persistent)
    - content: 메모리 텍스트
    - metadata: memory_type, importance, tags, created_at 등
    """

    def __init__(
        self,
        db_dir: str | Path = "./memory_db",
        collection_name: str = "long_term_memory",
        embedding_model: Optional[EmbeddingModel] = None,
    ) -> None:
        self.db_dir = Path(db_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model or EmbeddingModel()

        self.client = chromadb.PersistentClient(
            path=str(self.db_dir),
            settings=Settings(allow_reset=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
        )

    # -----------------------------
    # 메모리 쓰기
    # -----------------------------
    def add_memory(
        self,
        content: str,
        memory_type: str = "episodic",  # "profile" | "episodic" | "knowledge"
        importance: int = 3,
        tags: Optional[List[str]] = None,
    ) -> str:
        mem_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat(timespec="seconds")

        emb = self.embedding_model.embed([content])

        metadata = {
            "memory_type": memory_type,
            "importance": int(importance),
            "tags": tags or [],
            "created_at": now,
        }

        self.collection.add(
            documents=[content],
            embeddings=emb,
            ids=[mem_id],
            metadatas=[metadata],
        )
        return mem_id

    # -----------------------------
    # 메모리 검색
    # -----------------------------
    def search_memories(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query_emb = self.embedding_model.embed([query])

        where: Dict[str, Any] = {}
        if memory_type:
            # 특정 타입만 필터링 (예: profile 메모리만)
            where["memory_type"] = memory_type

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        out: List[Dict[str, Any]] = []
        for text, meta, dist in zip(docs, metas, dists):
            out.append(
                {
                    "content": text,
                    "metadata": meta,
                    "distance": float(dist),
                }
            )
        return out
