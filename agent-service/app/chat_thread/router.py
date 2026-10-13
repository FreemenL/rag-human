from fastapi import APIRouter, Depends, HTTPException, Response, status,Request
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from app.agnet.service import get_histrory_message
from app.chat_thread.crud import ChatThreadService
from app.chat_thread.exceptions import ChatThreadNotFound
from app.chat_thread.schemas import ChatThreadResponse, ChatThreadCreate, ChatHistoryResponse
from app.core.database import get_db
from app.core.logging import get_logger
from langgraph.graph.state import CompiledStateGraph
logger = get_logger(__name__)

user_id = 1 # 先用假数据
# 定义router
router = APIRouter(prefix="/chat-threads")

# @router.get("/hello")
# async def say_hello()->str:
#     return "hi,安心保欢迎你！"

# # 定义依赖注入函数，返回值是ChatThreadService，而这个函数又依赖get_db这个函数来获取session
def get_thread_service(db: AsyncSession = Depends(get_db)) -> ChatThreadService:
    return ChatThreadService(db)

def get_agent(request: Request):
    return  request.app.state.agent

# # 定义接口
@router.post("", response_model=ChatThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_thread(
    body: ChatThreadCreate,
    chat_thread_service: ChatThreadService = Depends(get_thread_service)
) -> ChatThreadResponse:
    logger.info("body")
    """创建新会话"""
    thread = await chat_thread_service.create_chat(user_id, body.title)
    logger.info("会话创建成功~", thread_id=str(thread.id), user_id=user_id)
    return thread
@router.get("", response_model=list[ChatThreadResponse], status_code=status.HTTP_200_OK)
async def get_chat_list(
    chat_thread_service: ChatThreadService = Depends(get_thread_service)
) -> list[ChatThreadResponse]:
    """查询当前用户的会话列表"""
    thread = await chat_thread_service.list_by_user(user_id)
    logger.info(f"查询用户{user_id}的会话列表，共查询到{len(thread)}条~")
    return thread

@router.patch("/{thread_id}")
async def rename_thread(
    thread_id: uuid.UUID,
    body: ChatThreadCreate,
    chat_thread_service: ChatThreadService = Depends(get_thread_service)
) -> ChatThreadResponse:
    """rename"""
    try:
        result = await chat_thread_service.rename(thread_id, user_id, body.title)
        logger.info(f"重命名会话成功～", thread_id=str(thread_id), user_id=user_id)
        return result
    except ChatThreadNotFound as ex:
        raise _not_found() from ex

@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(
    thread_id: uuid.UUID,
    chat_thread_service: ChatThreadService = Depends(get_thread_service)
) :
   try:
       """删除"""
       await chat_thread_service.delete(thread_id, user_id)
       logger.info(f"删除会话成功～", thread_id=str(thread_id), user_id=user_id)
   except ChatThreadNotFound as ex:
       raise _not_found() from ex

@router.get("/{thread_id}/messages")
async def get_messages_history(
    thread_id: uuid.UUID,
    agent:CompiledStateGraph = Depends(get_agent),
    chat_thread_service: ChatThreadService = Depends(get_thread_service)
):
    await chat_thread_service.required_owner(thread_id,user_id)
    messages = await get_histrory_message(agent,thread_id)
    return ChatHistoryResponse(thread_id=thread_id,messages=messages)


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="chat chat_thread not found")