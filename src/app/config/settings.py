# src/app/config/settings.py
from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import BaseModel

# 현재 작업 디렉토리(프로젝트 루트)의 .env 를 읽는다.
# (항상 루트에서 `python -m src.main` 형태로 실행한다고 가정)
load_dotenv()


class Settings(BaseModel):
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.2

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

        return cls(
            openai_api_key=api_key,
            openai_model=model,
            openai_temperature=temperature,
        )


# 전역에서 import 해서 쓰는 설정 인스턴스
settings = Settings.from_env()
