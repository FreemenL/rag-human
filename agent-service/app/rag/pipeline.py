from __future__ import annotations

from app.core.config import settings
from app.rag.models import KnowledgeDocument
from app.rag.crud import PostgresKnowledgeRepository
from app.rag.parsing import MinerUParser
from app.rag.optimizing import MarkdownOptimizer
from app.rag.chunking import MarkdownChunker
from app.rag.embedding import BGEM3Embedding
from app.rag.vector_store import MilvusVectorStore
from app.core.logging import get_logger

logger = get_logger(__name__)


class KnowledgeBuildPipeline:
    """按固定顺序编排知识库构建步骤。"""

    def __init__(
        self,
        repository: PostgresKnowledgeRepository,
        vector_store: MilvusVectorStore,
        embedding: BGEM3Embedding
    ) -> None:
        # 初始化管道所需的组件
        self.parser = MinerUParser(token=settings.mineru_token)
        self.optimizer = MarkdownOptimizer()
        self.chunker = MarkdownChunker()
        self.embedding = embedding
        self.repository = repository
        self.vector_store = vector_store
        logger.info("knowledge pipeline 初始化完成")

    async def ingest_document(
        self,
        document: KnowledgeDocument,
    ) -> bool:
        try:
            # 1. 查询文档
            exists = await self.repository.is_document_exists(document.clause_id)
            logger.info(f'准备处理文档: 【{document.document_name}】, clause_id: {document.clause_id}')
            logger.info(f'document.source_uri:{document.source_uri}')
            # 2. 解析Markdown。
            parsed_markdown = self.parser.parse(document.source_uri)
            logger.info(f'文档解析已完成: 【{document.document_name}】')

            # 3. 优化文档
            optimized_markdown = await self.optimizer.optimize(parsed_markdown)
            logger.info(f'文档优化已完成: 【{document.document_name}】')

            # 4. 切分文档
            parent_chunks, child_chunks = self.chunker.chunk(
                markdown=optimized_markdown,
                clause_id=document.clause_id,
            )
            logger.info(f'''
                文档分块已完成: 【{document.document_name}】. 
                parent_chunks: {len(parent_chunks)}, child_chunks: {len(child_chunks)}
            ''')

            # 5. PostgreSQL 保存可管理、可追溯的原文和分块。
            await self.repository.save_all(exists, document, parent_chunks, child_chunks)
            logger.info(f'文档保存已完成: 【{document.document_name}】')

            # 6. 向量化
            vectors = self.embedding.embed_documents(
                [chunk.content for chunk in child_chunks]
            )
            logger.info(f'文档向量生成已完成: 【{document.document_name}】')

            # 7.向量入库
            if exists is not None:
                self.vector_store.delete_by_clause_id(document.clause_id)
            self.vector_store.store(
                document.document_name,
                child_chunks,
                vectors,
                document.product_ids,
            )

            return True
        except RuntimeError:
            logger.exception(f'处理文档失败，【{document.document_name}】')
            return False