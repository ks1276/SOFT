# src/app/config/settings.py
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


# 프로젝트 루트 디렉토리 (SOFT)
BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseModel):
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.2

    # --- Google 검색 ---
    google_search_api_key: str | None = None
    google_search_cx: str | None = None

    # --- RAG 설정 ---
    rag_pdf_dir: Path
    rag_db_dir: Path
    rag_collection_name: str = "course_rag"
    rag_embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 가 .env(또는 환경변수)에 없습니다. "
                "프로젝트 루트의 .env 파일을 확인하세요."
            )

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

        google_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        google_cx = os.getenv("GOOGLE_SEARCH_CX")

        # .env 에서 경로 override 가능, 없으면 기본값 사용
        rag_pdf_dir_env = os.getenv("RAG_PDF_DIR")
        rag_db_dir_env = os.getenv("RAG_DB_DIR")

        rag_pdf_dir = Path(rag_pdf_dir_env) if rag_pdf_dir_env else BASE_DIR / "data" / "pdfs"
        rag_db_dir = Path(rag_db_dir_env) if rag_db_dir_env else BASE_DIR / "data" / "chroma_rag"

        emb_model_name = os.getenv(
            "RAG_EMBEDDING_MODEL",
            "paraphrase-multilingual-MiniLM-L12-v2",
        )

        return cls(
            openai_api_key=api_key,
            openai_model=model,
            openai_temperature=temperature,
            google_search_api_key=google_key,
            google_search_cx=google_cx,
            rag_pdf_dir=rag_pdf_dir,
            rag_db_dir=rag_db_dir,
            rag_collection_name=os.getenv("RAG_COLLECTION_NAME", "course_rag"),
            rag_embedding_model_name=emb_model_name,
        )


settings = Settings.from_env()

# --- Memory 설정 ---
memory_db_dir: Path
memory_collection_name: str = "course_memory"
