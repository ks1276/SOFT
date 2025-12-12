# rag/loader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

from pypdf import PdfReader


@dataclass
class TextChunk:
    id: str
    text: str
    metadata: Dict[str, Any]


def read_pdf_text(pdf_path: Path) -> str:
    """단일 PDF에서 모든 페이지 텍스트를 합쳐서 반환"""
    reader = PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    아주 단순한 청크 분리기:
    - 글자 단위로 자름
    - overlap을 주어서 문맥이 조금씩 겹치게
    """
    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - chunk_overlap  # overlap 만큼 뒤로
    return chunks


def load_pdfs_from_dir(
    pdf_dir: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> List[TextChunk]:
    """
    pdf_dir 안의 모든 PDF 파일을 읽어서 TextChunk 리스트로 반환
    id 형식: {파일이름}_{chunk_index}
    metadata: {"source": 파일명, "chunk_index": i}
    """
    pdf_dir = Path(pdf_dir)
    chunks: List[TextChunk] = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        raw_text = read_pdf_text(pdf_path)
        if not raw_text.strip():
            continue

        split_chunks = split_text(
            raw_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for i, c in enumerate(split_chunks):
            cid = f"{pdf_path.stem}_{i}"
            meta = {
                "source": pdf_path.name,
                "chunk_index": i,
            }
            chunks.append(TextChunk(id=cid, text=c, metadata=meta))

    return chunks
