from __future__ import annotations

from dataclasses import replace
from app.core.config import settings
from app.rag.embedding import BGEM3Embedding
from app.rag.crud import PostgresKnowledgeRepository
from app.rag.vector_store import RetrievedChunk, MilvusVectorStore
from app.core.logging import get_logger
logger = get_logger(__name__)

class KnowledgeRetriever:
    """编排召回、重排和父块回查。"""

    output_fields = [
        "clause_id",
        "parent_id",
        "document_name",
        "section_path",
        "content",
    ]

    def __init__(
        self,
        rerank_model: str,
        *,
        embedding: BGEM3Embedding,
        vector_store: MilvusVectorStore,
        repository: PostgresKnowledgeRepository,
    ) -> None:
        # 如果开启rerank功能，再导入rerank组件
        if settings.rerank_enable:
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker(rerank_model)

        self.embedding = embedding
        self.vector_store = vector_store
        self.repository = repository
        logger.info("knowledge retrieval 初始化完成")

    async def search(self, query:str, product_id: int, top_k:int) -> list[RetrievedChunk]:
        # 1.问题向量化
        vectors = self.embedding.embed_query(query)
        # 2.检索Milvus
        chunks = self.vector_store.search(
            vectors,
            top_k,
            filter_expression=_build_filter(product_id), # ARRAYS_CONTAINS(product_ids, 1)
            output_fields=self.output_fields,
        )
        # 3.重排
        reranked:list[RetrievedChunk] = self.rerank(query, chunks)[: top_k]

        # 4.同一父块命中多个子块时，只返回一次完整父块，减少重复上下文。
        parent_counts: dict[str, int] = {}
        # 统计子块的parent_id出现次数
        for chunk in reranked:
            parent_counts[chunk.parent_id] = (
                parent_counts.get(chunk.parent_id, 0) + 1
            )
        # 找出parent_id > 1的块
        repeated_parent_ids = [
            parent_id
            for parent_id, count in parent_counts.items()
            if count > 1
        ]
        # 查询父块
        parents = await self.repository.get_parent_chunks(repeated_parent_ids)

        # 5.封装为最终结果返回（查询出父块的子块移除，用父块作为结果）
        evidence: list[RetrievedChunk] = []
        added_parent_ids: set[str] = set() # 记录已添加的parent_id
        for chunk in reranked:
            # 如果parent已经记录过，跳过
            if chunk.parent_id in added_parent_ids:
                continue
            added_parent_ids.add(chunk.parent_id)
            # 编号递增

            chunk.source_id = f"ref-{(len(evidence) + 1):03d}"
            # 尝试获取查询到的父块
            parent = parents.get(chunk.parent_id, None)
            if parent is None:
                # 如果没有，说明是无重复的子块，则直接添加到最终结果
                evidence.append(chunk)
            else:
                # 如果有，说明是重复子块，用父块内容代替
                chunk.content = parent.content
                evidence.append(chunk)
        return evidence

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if not settings.rerank_enable or self.reranker is None:
            # 关闭Reranker
            return chunks
        # 重排打分
        scores = self.reranker.compute_score([[query, chunk.content] for chunk in chunks])
        if not isinstance(scores, list):
            scores = [float(scores)]
        # 用新得分替换旧得分
        reranked = [
            replace(chunk, score=float(score))
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        # 排序
        return sorted(reranked, key=lambda chunk: chunk.score, reverse=True)

def _build_filter(product_id: int | None) -> str | None:
    if product_id is None:
        return None
    return f"ARRAY_CONTAINS(product_ids, {product_id})"