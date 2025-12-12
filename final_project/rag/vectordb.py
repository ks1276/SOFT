# rag/vectordb.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .loader import load_pdfs_from_dir, TextChunk
from .embedder import EmbeddingModel


class RAGVectorStore:
    """
    - PDF 디렉토리 → 청크 → 임베딩 → Chroma Persistent 컬렉션에 저장
    - 쿼리 → 임베딩 → ANN 검색 → 상위 k개 문서 반환
    """

    def __init__(
        self,
        db_dir: str | Path = "./chroma_db",
        collection_name: str = "project_docs",
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

    # -----------------------
    # 색인 구축
    # -----------------------
    def build_from_pdf_dir(
        self,
        pdf_dir: str | Path,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        reset: bool = False,
    ) -> int:
        """
        pdf_dir 안 PDF들을 읽어서 전체 컬렉션을 다시 구성.
        reset=True면 기존 컬렉션을 비우고 새로 생성.
        반환: 색인된 chunk 개수
        """
        if reset:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(self.collection_name)

        chunks: List[TextChunk] = load_pdfs_from_dir(
            pdf_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        ids = [c.id for c in chunks]
        metadatas = [c.metadata for c in chunks]
        embeddings = self.embedding_model.embed(texts)

        # Chroma add
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

        return len(chunks)

    # -----------------------
    # 검색
    # -----------------------
    def query(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        query_text와 가장 유사한 문서 청크 top_k개 반환
        반환 형식: [{ "text": ..., "metadata": ..., "distance": ... }, ...]
        """
        query_emb = self.embedding_model.embed([query_text])

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        out: List[Dict[str, Any]] = []
        for text, meta, dist in zip(docs, metas, dists):
            out.append(
                {
                    "text": text,
                    "metadata": meta,
                    "distance": float(dist),
                }
            )
        return out
