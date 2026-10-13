from __future__ import annotations

from typing import Any
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class BGEM3Embedding:
    """加载一次 BGE-M3，同时生成稠密向量和稀疏向量。"""

    def __init__(
            self,
            *,
            device: str | None = None,
            model_name: str | None = None,
    ) -> None:
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction
        logger.info("BGEM3Embedding initialize start")
        # 优先级：显式入参 > .env 中的本地路径 > HuggingFace repo id。
        # 传本地路径时 FlagEmbedding 会跳过 snapshot_download，不访问网络。
        model_name = model_name or settings.embedding_model_path or "BAAI/bge-m3"
        # 模型权重较大，因此保存在对象中，由整个应用复用。
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            normalize_embeddings=True,
            use_fp16=False,
            return_dense=True,# 是否返回稠密向量
            return_sparse=True,# 是否返回稀疏向量
            return_colbert_vecs=False,
        )

        logger.info("BGEM3Embedding initialized")

    def dim(self) -> int:
        return self.embedding_function.dim['dense']

    def embed_query(self, text: str) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("query must not be empty")
        result = self.embedding_function([text])
        return {'dense': result['dense'][0], 'sparse': result['sparse'][0:1]}

    def embed_documents(self, texts: list[str]) -> dict[str, Any]:
        if not texts:
            raise ValueError("texts must not be empty")
        return self.embedding_function(texts)