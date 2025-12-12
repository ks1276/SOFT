# rag/embedder.py
from __future__ import annotations
from typing import List

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """
    멀티링궐 문장 임베딩 래퍼
    - 기본 모델: paraphrase-multilingual-MiniLM-L12-v2
      (영/한 모두 지원, 비교적 가벼움)
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        normalize: bool = True,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def embed(self, texts: List[str]) -> List[List[float]]:
        emb = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        if isinstance(emb, np.ndarray):
            return emb.tolist()
        return emb
