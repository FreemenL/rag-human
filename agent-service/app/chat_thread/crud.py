# from app.chat_thread.schemas import ChatThreadCreate
from app.chat_thread.exceptions import ChatThreadNotFound
from app.chat_thread.models import ChatThreadTable
# from app.core.database import get_db
from sqlalchemy import select, desc, update,func
from sqlalchemy.ext.asyncio import AsyncSession, result
from app.core.logging import get_logger
logger = get_logger(__name__)

class ChatThreadService:
    def __init__(self, session: AsyncSession):
        self.session:AsyncSession = session

    async def create_chat(self,user_id: int,title: str)->ChatThreadTable :
        async with self.session.begin():
            chat_thread = ChatThreadTable(user_id=user_id, title=title)
            self.session.add(chat_thread)
            return chat_thread

    async def list_by_user(self,user_id: int)->list[ChatThreadTable]:
        # 执行查询
        result = await self.session.execute(
            # select * from chat_threads where user_id = ?
            select(ChatThreadTable)
            .where(ChatThreadTable.user_id == user_id)
            .order_by(
                desc(ChatThreadTable.updated_at),
                desc(ChatThreadTable.created_at)
            )
        )
        return result.scalars().all()

    async def rename(self,thread_id,user_id,title)->ChatThreadTable:
        """重命名会话"""
        async with self.session.begin():
            # 尝试更新title
            result = await self.session.execute(
                update(ChatThreadTable)
                .values(title=title)
                .where(ChatThreadTable.id == thread_id, ChatThreadTable.user_id == user_id)
                .returning(ChatThreadTable)
            )
            # 判断结果 是否更新成功
            thread = result.scalar_one_or_none()
            if thread is None:
                raise ChatThreadNotFound(f"更新失败thread_id{thread_id}错误！")
        return thread

    async def delete(self,thread_id,user_id)->ChatThreadTable:
        async with self.session.begin():
            thread = await self.session.get(ChatThreadTable,thread_id)
            if not thread or thread.user_id != user_id:
                raise ChatThreadNotFound(f"删除失败thread_id{thread_id}错误！")
            await self.session.delete(thread)

    async def required_owner(self,thread_id,user_id):
       thread = await self.session.get(ChatThreadTable,thread_id)
       if not thread or thread.user_id != user_id:
           raise ChatThreadNotFound(f"非法访问，thread_id{thread_id}错误")
       return thread

    async def touch(self, thread_id):
        async with self.session.begin():
            await self.session.execute(
                update(ChatThreadTable)
                .values(updated_at=func.now())
                .where(ChatThreadTable.id == thread_id)
            )
