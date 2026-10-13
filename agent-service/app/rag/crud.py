from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import insert, select, delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .models import (
    DocumentTable,
    ChildChunkTable,
    ParentChunkTable,
    KnowledgeChunk,
    KnowledgeDocument,
)

class PostgresKnowledgeRepository:
    """Persist RAG documents, parent chunks, and child chunks in PostgreSQL."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory


    async def is_document_exists(self, clause_id: int) -> bool:
        async with self.session_factory() as session:
            document = await session.get(DocumentTable, clause_id)
        return document is not None

    async def save_all(
        self,
        exists: bool,
        document:KnowledgeDocument,
        parent_chunks: list[KnowledgeChunk],
        child_chunks: list[KnowledgeChunk]
    ):
        async with self.session_factory() as session:
            async with session.begin():
                if exists:
                    # 可能有旧值，需要先清空表
                    await session.execute(delete(ParentChunkTable).where(ParentChunkTable.clause_id == document.clause_id))
                    await session.execute(delete(ChildChunkTable).where(ChildChunkTable.clause_id == document.clause_id))
                # 处理Document
                await session.merge(_to_document_table(document))
                await _insert_parent_chunks(session, parent_chunks)
                await _insert_child_chunks(session, child_chunks)

    async def get_parent_chunks(
        self,
        parent_ids: list[str],
    ) -> dict[str, KnowledgeChunk]:
        """根据id查询父片段"""
        if not parent_ids:
            return {}
        query_ids = [UUID(str(parent_id)) for parent_id in parent_ids]
        async with self.session_factory() as session:
            result = await session.execute(
                select(ParentChunkTable).where(
                    ParentChunkTable.id.in_(query_ids)
                )
            )
            rows = result.scalars().all()
        chunks = [KnowledgeChunk.model_validate(row) for row in rows]
        return {str(chunk.id): chunk for chunk in chunks}

def _to_document_table(document: KnowledgeDocument) -> DocumentTable:
    return DocumentTable(
        clause_id=document.clause_id,
        document_name=document.document_name,
        source_uri=document.source_uri
    )

async def _insert_parent_chunks(session: AsyncSession, chunks: list[KnowledgeChunk]) -> None:
    if not chunks:
        return
    await session.execute(
        insert(ParentChunkTable),
        [
            {
                "id": chunk.id,
                "clause_id": chunk.clause_id,
                "content": chunk.content,
                "section_path": chunk.section_path,
            }
            for chunk in chunks
        ],
    )


async def _insert_child_chunks(session: AsyncSession, chunks: list[KnowledgeChunk]) -> None:
    if not chunks:
        return
    await session.execute(
        insert(ChildChunkTable),
        [
            {
                "id": chunk.id,
                "parent_id": chunk.parent_id,
                "clause_id": chunk.clause_id,
                "section_path": chunk.section_path,
                "content": chunk.content,
            }
            for chunk in chunks
        ],
    )

def _parent_chunk_from_row(row: Any) -> KnowledgeChunk:
    return KnowledgeChunk.model_validate(row)