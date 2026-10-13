from fastapi import APIRouter,Request,HTTPException
from fastapi.params import Depends
from fastapi.sse import EventSourceResponse,ServerSentEvent
# from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.state import CompiledStateGraph
# from openai.types.responses import response

from app.agnet.schemas import ChatRequest
from app.agnet.service import chat_stream
from app.security.jwt import get_current_user, UserContext
from app.chat_thread.crud import ChatThreadService
from app.core.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chat")

def get_agent(request: Request):
    return  request.app.state.agent

def get_thread_service(db: AsyncSession = Depends(get_db)) -> ChatThreadService:
    return ChatThreadService(db)

@router.post("",response_class=EventSourceResponse)
async def chat(
        body: ChatRequest,
        agent:CompiledStateGraph = Depends(get_agent),
        chat_thread_service: ChatThreadService = Depends(get_thread_service),
        user_info: UserContext = Depends(get_current_user)
):
        message = body.message
        decision = body.decision

        if message is None and decision is None:
            raise  HTTPException(status_code=400,detail="Message or decision cannot be None")
        if body.thread_id is None:
            raise HTTPException(status_code=400,detail="thread_id cannot be None")

        # 更新会话的update_at时间
        await chat_thread_service.touch(body.thread_id)

        async for chunk in chat_stream(agent,body.thread_id,message,decision,user_info):
            yield ServerSentEvent(event=chunk['event'],data=chunk['data'])

