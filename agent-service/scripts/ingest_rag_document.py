"""RAG 知识库构建脚本 —— 从 biz-service 拉取产品与条款，逐文档构建向量知识库。

用法:
    cd agent-service
    python scripts/ingest_rag_document.py [--clear]

选项:
    --clear   构建前先清空 Milvus Collection（默认不清空，增量更新）
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# 确保项目根目录（agent-service/）在 sys.path 中，必须在导入 app.* 之前执行
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from app.rag.crud import PostgresKnowledgeRepository
from app.rag.embedding import BGEM3Embedding

from app.clients.biz_client import BusinessServiceClient
from app.core.config import settings
from app.core.database import init_db, close_db, get_session_factory
from app.core.logging import get_logger
from app.rag.models import KnowledgeDocument
from app.rag.pipeline import KnowledgeBuildPipeline
from app.rag.vector_store import MilvusVectorStore

logger = get_logger(__name__)


async def fetch_all_products(client: BusinessServiceClient) -> list[dict]:
    """分页拉取所有产品。"""
    products: list[dict] = []
    page = 1
    page_size = 50
    while True:
        resp = await client.list_products(page=page, page_size=page_size)
        items: list[dict] = resp.get("items", [])
        if not items:
            break
        products.extend(items)
        if len(items) < page_size:
            break
        page += 1
    return products


async def collect_documents(
        client: BusinessServiceClient,
        products: list[dict],
) -> list[KnowledgeDocument]:
    """遍历产品查询主条款，按 clause_id 去重，组装 KnowledgeDocument 列表。

    biz-service 返回的 file_path 形如 ``data/raw/kb/xxx.pdf``，相对于
    biz-service 容器的 /app 目录。宿主机上实际路径为 ``{项目根}/data/raw/kb/xxx.pdf``。
    """
    # clause_id -> {"clause": {...}, "product_ids": [...]}
    clause_map: dict[int, dict] = {}

    for p in products:
        product_id: int = p["id"]
        try:
            resp = await client.list_clauses_of_product(product_id=product_id)
        except Exception:
            logger.exception("查询产品 %s 的条款失败，跳过", product_id)
            continue

        for c in resp.get("data", []):
            if c.get("clause_type") != "main_clause":
                continue
            cid: int = c["id"]
            if cid in clause_map:
                clause_map[cid]["product_ids"].append(product_id)
            else:
                clause_map[cid] = {
                    "clause": c,
                    "product_ids": [product_id],
                }

    documents: list[KnowledgeDocument] = []
    data_root = _PROJECT_ROOT.parent  # hm-insurance/
    for cid, info in clause_map.items():
        c = info["clause"]
        file_path: str = c.get("file_path", "")
        if not file_path:
            logger.warning("条款 %s 缺少 file_path，跳过", cid)
            continue
        source_uri = str(data_root / file_path)
        doc_name = Path(file_path).stem
        documents.append(
            KnowledgeDocument(
                clause_id=cid,
                document_name=doc_name,
                source_uri=source_uri,
                product_ids=info["product_ids"],
            )
        )

    return documents


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 RAG 知识库")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="构建前先清空 Milvus Collection",
    )
    args = parser.parse_args(argv)

    # 1. 初始化数据库
    await init_db()
    repository = PostgresKnowledgeRepository(get_session_factory())

    # 2. 初始化 biz-client
    client = BusinessServiceClient()
    await client.init()

    # 3.初始化embedding
    embedding = BGEM3Embedding()

    # 4.初始化Milvus
    vs = MilvusVectorStore(
        uri=settings.milvus_uri,
        collection_name=settings.milvus_collection,
        dense_dim=embedding.dim()
    )

    try:
        # 3. 可选：清空 Milvus Collection
        if args.clear:
            logger.warning("正在清空 Milvus Collection: %s", settings.milvus_collection)
            if vs.client.has_collection(vs.collection_name):
                vs.client.drop_collection()
                logger.info("Milvus Collection 已清空")
            else:
                logger.info("Milvus Collection 不存在，跳过清空")

        # 4. 拉取产品
        logger.info("正在拉取产品列表...")
        products = await fetch_all_products(client)
        logger.info("共获取 %d 个产品", len(products))

        if not products:
            logger.warning("未获取到任何产品，退出")
            return 0

        # 5. 收集条款文档
        logger.info("正在查询产品条款...")
        documents = await collect_documents(client, products)
        logger.info("去重后共 %d 份主条款文档待处理", len(documents))

        if not documents:
            logger.warning("未获取到任何条款文档，退出")
            return 0

        # 6. 逐文档构建知识库
        pipeline = KnowledgeBuildPipeline(repository, vs, embedding)

        success_count = 0
        for i, doc in enumerate(documents, 1):
            logger.info("[%d/%d] 开始处理: %s", i, len(documents), doc.document_name)
            ok = await pipeline.ingest_document(doc)
            if ok:
                success_count += 1
                logger.info("[%d/%d] 处理成功: %s", i, len(documents), doc.document_name)
            else:
                logger.error("[%d/%d] 处理失败: %s", i, len(documents), doc.document_name)

        logger.info(
            "知识库构建完成: %d/%d 成功",
            success_count,
            len(documents),
        )
        return 0 if success_count == len(documents) else 1

    finally:
        await client.close()
        await close_db()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))