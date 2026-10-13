from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from app.agnet.recommendation.graph import Recommendation
from app.agnet.models import ChatRuntimeContext, ParentState, Intent
from app.agnet.intent_indentify import intent_router
from app.core.logging import get_logger
from app.agnet.chitchat import Chitchat
logger = get_logger(__name__)

async def _route_node(state: ParentState) -> Intent:
    """意图识别节点"""
    # 1.获取最近一条AI回复消息，作为意图识别上下文
    previous_ai_message = _previous_ai_message(state['messages'])
    payload = {}
    if previous_ai_message:
        payload["conversation_history"] = [
            {"role": "assistant", "content": previous_ai_message}
        ]
    # 2.尝试意图识别，传入用户问题、上下文
    result = intent_router.route(message = state['messages'][-1].content, payload = payload)
    previous_workflow = state.get('active_workflow', '')
    # 3.判断识别结果，如果是str，就是寒暄，直接结束，否则就进入工作流
    if result and isinstance(result, str):
        return Command(
            update={
                'messages': [AIMessage(content=result)],
                'previous_workflow': previous_workflow,
                'active_workflow': 'chitchat'
            },
            goto=END  # 跳转到 intent 识别的节点
        )
    # 4.跳转到识别出的节点，并更新state状态
    return Command(
        update={
            'intent': result,
            'previous_workflow': previous_workflow,
            'active_workflow': result.intent
        },  # 更新intent到父State
        goto=result.intent # 跳转到 intent 识别的节点
    )

async def _chitchat_node(state: ParentState):
    """闲聊节点"""
    return {
        "messages": [AIMessage(content="你好，今天聊点啥呢？")],
    }

async def _claim_node(state: ParentState):
    """理赔咨询、引导节点"""
    return {
        "messages": [AIMessage(content="暂时无法理赔")],
    }

def _fallback_node(state: ParentState) -> ParentState:
    answer = "我还不能确定你的需求。你可以说明是想咨询产品条款、做保险方案、试算保费，还是需要投保流程帮助。"
    return {
        "messages": [AIMessage(content=answer)],
    }

def _human_handoff(state: ParentState) -> ParentState:
    return {
        "messages": [AIMessage(content="已记录转人工请求，后续可接入人工客服队列或工单系统。")],
    }



async def create_agent(checkpointer):
    # 3.创建Graph
    builder = StateGraph(ParentState, context_schema=ChatRuntimeContext)
    builder.add_node('route', _route_node)
    builder.add_node('chitchat', Chitchat().build_graph())
    builder.add_node('recommendation_plan', Recommendation().build_graph())
    builder.add_node('claim', _claim_node)
    builder.add_node('fallback', _fallback_node)
    builder.add_node('human_handoff', _human_handoff)


    builder.add_edge(START, 'route')
    builder.add_edge('chitchat', END)
    builder.add_edge('recommendation_plan', END)
    builder.add_edge('claim', END)
    builder.add_edge('fallback', END)
    builder.add_edge('human_handoff', END)

    graph = builder.compile(checkpointer=checkpointer)

    return graph

def _previous_ai_message(messages: list[BaseMessage]) -> str | None:
    """返回本轮用户消息之前最近一条可展示的 AI 文本。"""
    for item in reversed(messages):
        if not isinstance(item, AIMessage):
            continue
        if item.tool_calls or item.additional_kwargs.get("tool_calls"):
            continue
        content = item.content
        if content:
            return content
    return None