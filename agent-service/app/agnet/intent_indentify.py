from enum import StrEnum

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

from app.agnet.models import RouteResult
import re
from app.core.logging import get_logger
import os
from dotenv import load_dotenv
logger = get_logger(__name__)
load_dotenv()
class IntentRouter:

    def __init__(self):
        # 初始化模型

        model = init_chat_model(
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
        # 结构化绑定;打上内部tag,便于流式输出时过滤掉意图识别的LLM结果
        self.model = model.with_structured_output(RouteResult).with_config(
            tags=["internal_intent_router"]
        )

    def route(self, message: str, payload: dict):
        # 1.寒暄识别（基于正则表达式、规则识别）
        intent = _match_social_intent(message)
        if intent:
            return SOCIAL_RESPONSES[intent]

        # 2.寒暄识别失败，走模型识别
        try:
            result = self.model.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"user_message: {message}, context：{payload}")
            ])
            logger.info(f"模型意图识别结果：{result}")
            return result
        except Exception:
            logger.exception("模型意图识别失败，切换至关键字意图识别")
            return RouteResult(intent="fallback", confidence=0.2, reason="模型意图识别失败")

# --------------------模型意图识别------------------
SYSTEM_PROMPT = (
    "你是保险商城智能客服的业务域路由节点，请根据用户message和上下文信息决定把用户消息交给哪个业务 Agent，"
    "intent只能从 recommendation_plan、claim、human_handoff、chitchat、fallback 中选择。"
    "保险产品咨询、保险产品推荐、保险产品组合方案管理、投保咨询归 recommendation_plan。"
    "理赔咨询、责任、材料、流程、进度、报案归 claim。"
    "高风险、投诉、用户明确要求人工归 human_handoff。"
    "与保险无关的开放聊天归 chitchat；无法判断归 fallback。"
)
# -------------------寒暄识别--------------------------
class SocialIntent(StrEnum):
    GREETING = "greeting"
    THANKS = "thanks"
    GOODBYE = "goodbye"

SOCIAL_PATTERNS = {
    SocialIntent.GREETING: (
        r"(你|您)?好(呀|啊|哦)?",
        r"嗨",
        r"哈喽",
        r"hello",
        r"hi",
        r"在吗",
    ),
    SocialIntent.THANKS: (
        r"谢谢(你|您)?",
        r"多谢(你|您)?",
        r"感谢(你|您)?",
        r"辛苦了",
    ),
    SocialIntent.GOODBYE: (
        r"再见",
        r"拜拜",
        r"先这样",
        r"没事了",
    ),
}

SOCIAL_RESPONSES = {
    SocialIntent.GREETING: "您好，我是你的专属保险顾问😃。可以帮你解决投保、保单或理赔问题。",
    SocialIntent.THANKS: "不客气，很高兴帮到您。后续有保险问题随时可以问我。",
    SocialIntent.GOODBYE: "好的，后续需要帮助随时联系我。",
}

def _normalize_message(message: str) -> str:
    """忽略常见空白和标点，只保留用于意图匹配的文本。"""
    return re.sub(r"[\s，。！？、,.!?~～]", "", message).lower()

def _match_social_intent(message: str) -> SocialIntent | None:
    normalized = _normalize_message(message)

    for intent, patterns in SOCIAL_PATTERNS.items():
        if any(re.fullmatch(pattern, normalized) for pattern in patterns):
            return intent
    return None


intent_router = IntentRouter()