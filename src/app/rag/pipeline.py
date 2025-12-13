# src/app/rag/pipeline.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from src.app.config.settings import settings


# ---------- globals ----------
_rag_client: Optional[PersistentClient] = None
_rag_embedder: Optional[SentenceTransformer] = None


def get_rag_db_dir() -> Path:
    # settings.rag_db_dir가 있으면 사용, 없으면 <BASE>/data/chroma_rag
    if hasattr(settings, "rag_db_dir"):
        return Path(getattr(settings, "rag_db_dir"))
    return Path.cwd() / "data" / "chroma_rag"


def get_rag_collection_name() -> str:
    return getattr(settings, "rag_collection_name", "course_rag")


def get_rag_client() -> PersistentClient:
    global _rag_client
    if _rag_client is None:
        db_dir = get_rag_db_dir()
        db_dir.mkdir(parents=True, exist_ok=True)
        _rag_client = chromadb.PersistentClient(path=str(db_dir))
    return _rag_client


def get_rag_collection():
    client = get_rag_client()
    return client.get_or_create_collection(
        name=get_rag_collection_name(),
        metadata={"hnsw:space": "cosine"},
    )


def get_rag_embedder() -> SentenceTransformer:
    global _rag_embedder
    if _rag_embedder is None:
        # 이미 settings에 멀티링구얼 임베더명이 있다고 했으니 재사용
        _rag_embedder = SentenceTransformer(settings.rag_embedding_model_name)
    return _rag_embedder


# ---------- PDF loader ----------
def _read_pdf_text(pdf_path: Path) -> str:
    """
    의존성 최소화를 위해 pypdf 사용 권장.
    requirements.txt: pypdf
    """
    from pypdf import PdfReader  # local import to avoid import-time failure

    reader = PdfReader(str(pdf_path))
    parts: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
    return "\n".join(parts)


# ---------- chunking ----------
def _clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []

    # 안전장치
    overlap = max(0, min(overlap, chunk_size - 1))
    
    chunks: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        j = min(n, i + chunk_size)
        chunk = text[i:j]
        if chunk:
            chunks.append(chunk)

        # ✅ 핵심: 끝까지 도달했으면 종료
        if j >= n:
            break

        # 다음 시작점(겹침 적용)
        i = j - overlap

        # ✅ 혹시라도 진행이 멈추는 경우 방어
        if i <= 0 and j == chunk_size:
            i = j  # 최소 전진 보장

    return chunks



# ---------- indexing ----------
def index_pdfs(pdf_dir: Path, rebuild: bool = False) -> Dict[str, Any]:
    """
    pdf_dir 내 PDF를 읽어서 청크→임베딩→Chroma에 저장
    """
    col = get_rag_collection()

    if rebuild:
        # 컬렉션을 비우고 재색인
        # (Chroma 버전에 따라 delete(where={}) 또는 recreate가 다를 수 있음)
        try:
            col.delete(where={})
        except Exception:
            pass

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        return {"ok": False, "message": f"No PDFs found in {pdf_dir}"}

    embedder = get_rag_embedder()

    total_chunks = 0
    for pdf_path in pdfs:
        raw = _read_pdf_text(pdf_path)
        chunks = _chunk_text(raw)

        if not chunks:
            continue

        # ids/metadata/documents 준비
        ids = [f"{pdf_path.name}::chunk::{k}" for k in range(len(chunks))]
        metadatas = [{"source": pdf_path.name, "chunk_index": k} for k in range(len(chunks))]
        embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

        # 중복 id가 있으면 add에서 에러가 날 수 있으므로, 간단히 upsert 권장
        try:
            col.upsert(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
        except Exception:
            # upsert 없는 구버전이면 add로 시도
            col.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)

        total_chunks += len(chunks)

    return {"ok": True, "pdf_count": len(pdfs), "chunk_count": total_chunks}


# ---------- query ----------
def query_rag(query: str, top_k: int = 5) -> Dict[str, Any]:
    col = get_rag_collection()
    qemb = get_rag_embedder().encode([query], show_progress_bar=False).tolist()

    res = col.query(
        query_embeddings=qemb,
        n_results=max(1, min(int(top_k), 10)),
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({
            "text": doc,
            "source": (meta or {}).get("source", ""),
            "chunk_index": (meta or {}).get("chunk_index", -1),
            "distance": float(dist) if dist is not None else None,
        })

    return {"query": query, "top_k": top_k, "hits": hits}


def format_rag_answer(result: Dict[str, Any]) -> str:
    hits = result.get("hits") or []
    if not hits:
        return "검색 결과가 없습니다."

    lines: List[str] = []
    for i, h in enumerate(hits[:5], 1):
        src = h.get("source", "")
        idx = h.get("chunk_index", -1)
        txt = (h.get("text") or "").strip().replace("\n", " ")
        lines.append(f"[{i}] ({src} / chunk {idx}) {txt[:300]}")
    return "\n".join(lines)
