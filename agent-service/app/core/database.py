# from __future__ import annotations
# from typing import AsyncGenerator
# from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
#
# from app.core.base import Base
# from app.core.config import settings
# from app.core.logging import get_logger
#
# logger = get_logger(__name__)
#
# if not settings.database_url:
#     raise RuntimeError("database_url未配置，初始化数据库失败！")
#
# # 1. 内部持有的引擎和会话工厂（不对外暴露，保持内聚）
# _engine = create_async_engine(settings.database_url, echo=False, pool_size=10, max_overflow=20)
# _async_session = async_sessionmaker(_engine, expire_on_commit=False)
#
# logger.info("数据库初始化成功")
#
# # 2. 依赖注入函数：获取数据库异步会话（供路由使用）
# async def get_db() -> AsyncGenerator[AsyncSession, None]:
#     async with _async_session() as session:
#         try:
#             yield session
#         finally:
#             await session.close()  # 👈 只负责关闭，不负责 commit/begin
#
#
# def get_session_factory() -> async_sessionmaker[AsyncSession]:
#     """提供给应用启动阶段组装 Repository，业务模块不直接访问私有变量。"""
#
#     return _async_session
#
# # 3. 统一管理：初始化数据库（创建表）
# async def init_db():
#     global _engine
#     global _async_session
#     # 初始化连接池和session工厂
#     _engine = create_async_engine(settings.database_url, echo=False, pool_size=10, max_overflow=20)
#     _async_session = async_sessionmaker(_engine, expire_on_commit=False)
#     # 自动建表
#     async with _engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)
#
# # 4. 统一管理：关闭数据库（销毁连接池）
# async def close_db():
#     await _engine.dispose()
#     logger.info("✅ 数据库连接已关闭")
from __future__ import annotations
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker, AsyncEngine

from app.core.base import Base
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

if not settings.database_url:
    raise RuntimeError("database_url未配置，初始化数据库失败！")

# 1. 内部持有的引擎和会话工厂（不对外暴露，保持内聚）
_engine: AsyncEngine = None
_async_session_factory = None

# 2. 统一管理：初始化数据库、自动建表
async def init_db():
    global _engine
    global _async_session_factory
    _engine = create_async_engine(settings.database_url, echo=False, pool_size=10, max_overflow=20)
    _async_session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    # 导入所有表模型，确保注册到 Base.metadata
    from app.chat_thread import models as _chat_thread_models  # noqa: F401
    from app.rag import models as _rag_models  # noqa: F401
    # 自动建表（已存在的表会自动跳过）
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ 数据库初始化成功")


# 3.定义一个函数，用来获取session_factory
def get_session_factory():
    return _async_session_factory

# 3. 依赖注入函数：获取数据库异步会话（供路由使用）
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with _async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


# 4. 统一管理：关闭数据库（销毁连接池）
async def close_db():
    await _engine.dispose()
    logger.info("✅ 数据库连接已关闭")