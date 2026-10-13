import time
from contextlib import asynccontextmanager
import structlog
from fastapi import FastAPI, Request
import uuid
from starlette.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.core.database import init_db,close_db
from app.chat_thread.router import router as chat_thread_router
from app.agnet.router import router as agent_router
from app.agnet.checkpointer import init_checkpointer,close_checkpointer
from app.agnet.graph import create_agent
from app.clients.biz_client import biz_client
from app.rag import init_rag_module, dispose_rag

# 初始化日志配置
configure_logging(level=settings.log_level, log_format=settings.log_format)
logger = get_logger(__name__)

# 生命周期管理（预留）
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("agent_service_starting ", app_version=settings.app_version)
        await init_db()
        # 初始化biz-service客户端
        await biz_client.init()
        # RAG模块资源
        await init_rag_module()
        # 初始化checkpointer
        checkpointer = await init_checkpointer()
        app.state.agent = await create_agent(checkpointer)

        yield
    finally:
        logger.info("agent_service_stopping")
        # 关闭rag
        await dispose_rag()
        await close_db()
        await close_checkpointer()
        # 关闭biz-service客户端
        await biz_client.close()
        logger.info("agent_service_stopped ")
# 初始化FastAPI
app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

# 日志输出中间件
@app.middleware("http")
async def add_log_uuid(request: Request, call_next):
    # 1. 清理并生成新的请求上下文
    structlog.contextvars.clear_contextvars()
    request_id = str(uuid.uuid4())

    # 2. 绑定当前请求的核心信息
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )

    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    # 3. 记录请求结束日志
    logger.info("request_processed", status_code=response.status_code, duration=f"{duration_ms:.4f}ms")
    return response

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins="http://127.0.0.1:5173",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_thread_router,prefix="/api/v1",tags=["会化管理相关接口"])
app.include_router(agent_router,prefix="/api/v1",tags=["agent对话相关接口"])

@app.get("/")
async def healthy():
    return {"status": "running", "port": "8001"}

if __name__ == "__main__":
    import uvicorn
    # 启动命令：python -m app.main
    config = uvicorn.Config(
        "app.main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=False
    )
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        # Ctrl+C 优雅退出：uvloop + asyncio.Runner 会把中断转成 KeyboardInterrupt 抛出，这里捕获避免打印堆栈
        logger.info("agent_service_stopped")