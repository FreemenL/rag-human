from langchain_core.messages import HumanMessage,AIMessage,BaseMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.stream import CustomTransformer
from langgraph.types import StateSnapshot

from app.agnet.models import ChatRuntimeContext
from app.chat_thread.schemas import ChatHistoryMessage
from app.core.logging import get_logger
from app.security.jwt import UserContext
from typing import Any

logger = get_logger(__name__)

async def chat_stream(agent:CompiledStateGraph,thread_id:str,message:str,decision:dict,user:UserContext):
    # 组织输入, 并且重置additional_info
    _input = {'messages': [HumanMessage(content=message)], "additional_info": None}
    # 会话记忆id
    config = _config_with_thread_id(thread_id)
    # 调用模型
    stream = await agent.astream_events(
        _input,
        config,
        context=ChatRuntimeContext(user_id=user.id, token=user.token),
        version='v3',
        transformers=[CustomTransformer]
    )
    additional_info = None
    # 返回结果
    async for event in stream:
        method = event['method']
        # 判断事件类型，可以是messages、interrupts等等
        if method == 'messages':
            data, metadata = event['params']['data']
            # 过滤意图识别节点内部 LLM 的输出（路由JSON），不推送给前端
            if "internal_intent_router" in (metadata.get("tags") or []):
                continue
            if isinstance(data, AIMessage):
                for chunk in data.content:
                    yield { "event": "message", "data": chunk}
            elif isinstance(data, dict) and data.get('delta') and data['delta'].get('text'):
                yield { "event": "message", "data": data['delta'].get('text', "")}
        if method == 'custom':
            data = event["params"]["data"]
            logger.info(custom_data=data)
            if isinstance(data, dict) and data.get("type") == "additional_info":
                additional_info = data.get("data")
    if additional_info:
        yield {
            "event": "additional_info",
            "data": additional_info,
        }
    # 消息结束，返回done
    yield {'event': 'done', 'data': {'thread_id': thread_id, 'finish': 'stop'}}

async def get_histrory_message(agent:CompiledStateGraph,thread_id:str)->list[ChatHistoryMessage]:
    # 记忆的thread_id
    config = {'configurable': {'thread_id': thread_id}}
    # 获取state的结束节点快照
    snapshots = [
        snapshot
        async for snapshot in agent.aget_state_history(config)
        if snapshot.next == ()
    ]
    # 反转顺序，最新快照放最后
    states = list(reversed(snapshots))

    # 处理消息
    return _project_history_from_snapshots(states)


def _message_text(content: str | list[str | dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {None, "text"}:
            parts.append(str(block.get("text", "")))
    return "".join(parts).strip()


def _project_history_from_snapshots(snapshots: list[StateSnapshot]) -> list[ChatHistoryMessage]:
    """从 LangGraph 完成态快照恢复可展示历史，并把每轮 additional_info 绑定到当轮 AI 回复。"""

    projected: list[ChatHistoryMessage] = []
    previous_visible_count = 0
    for snapshot in snapshots:
        values = snapshot.values if isinstance(snapshot.values, dict) else {}
        messages = _visible_messages(list(values.get("messages", [])))
        new_messages = messages[previous_visible_count:]
        additional_info = values.get("additional_info")
        last_assistant_index = _last_assistant_index(new_messages)
        for index, message in enumerate(new_messages):
            if (
                additional_info is not None
                and last_assistant_index is not None
                and index == last_assistant_index
                and message.role == "assistant"
            ):
                projected.append(message.model_copy(update={"additional_info": additional_info}))
            else:
                projected.append(message)
        previous_visible_count = len(messages)
    return projected

def _visible_messages(messages: list[BaseMessage]) -> list[ChatHistoryMessage]:
    return _project_history(messages)


def _last_assistant_index(messages: list[ChatHistoryMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "assistant":
            return index
    return None

def _project_history(messages: list[BaseMessage]) -> list[ChatHistoryMessage]:
    """把完整 LangGraph 消息投影为前端可展示的用户和最终助手文本。"""
    projected: list[ChatHistoryMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            if message.tool_calls or message.additional_kwargs.get("tool_calls"):
                continue
            role = "assistant"
        else:
            continue

        content = _message_text(message.content)
        if content:
            projected.append(ChatHistoryMessage(role=role, content=content))
    return projected

async def clear_message_history(agent: CompiledStateGraph, thread_id: str):
    await agent.checkpointer.adelete_thread(thread_id)

def _config_with_thread_id(thread_id):
    return {"configurable": {"thread_id": thread_id}}