from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph

from app.agnet.models import ParentState
from app.core.config import settings
from app.core.logging import get_logger
import os
from dotenv import load_dotenv
logger = get_logger(__name__)
load_dotenv()


SYSTEM_MESSAGE = """
你是保险智能客服，负责处理与保险业务无关的普通聊天。

要求：
1. 回复友好、自然，最多两句话。
2. 不假装有个人经历、情感、实时信息或外部能力。
3. 不回答医疗、法律、投资等高风险专业建议。
4. 不展开长期闲聊；每次回复都自然引导用户咨询保险产品、投保、保单或理赔。
5. 如果用户的问题本身需要实时信息或不属于你的能力范围，明确说明无法提供。
"""
MAX_CHIT_CHAT_COUNT = 3

class ChitchatState(ParentState):
    chitchat_count: int # 闲聊轮次

class Chitchat:

    def __init__(self):
        self.model =model = init_chat_model(
            model="qwen3.7-plus",
            model_provider="openai",
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            }
        )

    async def _handle_chitchat(
        self,
        state: ChitchatState,
    ) -> dict:
        """处理闲聊"""
        count = state.get('chitchat_count', 0)
        # 如果上一次不是闲聊，将count置为0
        if state.get('previous_workflow', '') != 'chitchat':
            count = 0
        logger.info(f'当前闲聊轮次：{count}')

        # 如果闲聊轮次达到上限，直接回复固定模版
        if count >= MAX_CHIT_CHAT_COUNT:
            return {'messages': [AIMessage('我主要协助处理保险产品、投保、保单和理赔咨询。 您可以直接告诉我想办理或了解的事项。')]}

        # 让模型生成闲聊内容
        response = await self.model.ainvoke([SystemMessage(content=SYSTEM_MESSAGE)] + state['messages'])
        # 更新回复和闲聊次数
        return {'messages': [response], 'chitchat_count': count+1}

    def build_graph(self):
        builder = StateGraph(ChitchatState)
        builder.add_node("handle_chitchat", self._handle_chitchat)
        # 设置Graph入口，等同于 add_edge(START, "handle_chitchat")
        builder.set_entry_point("handle_chitchat")
        # 设置Graph出口，等同于 add_edge("handle_chitchat", END)
        builder.set_finish_point("handle_chitchat")
        return builder.compile(checkpointer=True)