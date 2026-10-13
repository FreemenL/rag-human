"""保险知识库构建与检索模块。"""
from app.rag.models import KnowledgeDocument
from app.rag.pipeline import KnowledgeBuildPipeline
from app.rag.retrieval import KnowledgeRetriever
from app.rag.vector_store import MilvusVectorStore, RetrievedChunk
from app.rag.crud import PostgresKnowledgeRepository
from app.core.database import get_session_factory
from app.core.config import settings
from app.rag.embedding import BGEM3Embedding

_vector_store: MilvusVectorStore = None
_retriever: KnowledgeRetriever = None
_pipeline: KnowledgeBuildPipeline = None

async def init_rag_module():
    global _vector_store
    global _retriever
    global _pipeline

    # 向量模型
    _embedding = BGEM3Embedding()

    # 向量库
    _vector_store = MilvusVectorStore(
        uri=settings.milvus_uri,
        collection_name=settings.milvus_collection,
    )
    # 数据库
    _knowledge_repository = PostgresKnowledgeRepository(get_session_factory())

    # Retriever
    _retriever = KnowledgeRetriever(
        rerank_model=settings.rerank_model,
        embedding=_embedding,
        vector_store=_vector_store,
        repository=_knowledge_repository
    )

    # pipeline
    _pipeline = KnowledgeBuildPipeline(
        _knowledge_repository,
        _vector_store,
        _embedding
    )

def retriever() -> KnowledgeRetriever:
    global _retriever
    return _retriever

def pipeline() -> KnowledgeBuildPipeline:
    global _pipeline
    return _pipeline

async def dispose_rag():
    global _vector_store
    _vector_store.close()


__all__ = [
    'init_rag_module',
    'dispose_rag',
    'pipeline',
    'retriever',
    'KnowledgeRetriever',
    'RetrievedChunk',
    'KnowledgeBuildPipeline',
    'KnowledgeDocument',
]