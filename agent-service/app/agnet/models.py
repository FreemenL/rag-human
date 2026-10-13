from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NotRequired

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

# 意图识别常量
Intent = Literal[
    "chitchat",
    "recommendation_plan",
    "claim",
    "human_handoff",
    "fallback",
]

# 路由结果，通常路由结果必须说明判断原因和置信度
class RouteResult(BaseModel):
    intent: Intent = Field(description="The best target intent for the current user turn.")
    confidence: float = Field(ge=0, le=1, description="Routing confidence from 0 to 1.")
    reason: str = Field(description="判断用户意图的理由.")

# 父图state
class ParentState(MessagesState):
    route_info: NotRequired[RouteResult | None]  # 路由信息
    previous_workflow: NotRequired[str]  # 上一个工作流
    active_workflow: NotRequired[str]  # 当前激活的工作流
    additional_info: NotRequired[dict]  # 拓展信息

@dataclass
class ChatRuntimeContext:
    user_id: str | None
    token: str | None
