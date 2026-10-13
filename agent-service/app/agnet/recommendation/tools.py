
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from app.clients.biz_client import biz_client
from app.agnet.models import ChatRuntimeContext
from app.core.logging import get_logger
from app.rag import retriever, RetrievedChunk
from .models import InsuredProfile, InsuranceCategory
from langgraph.config import get_stream_writer
logger = get_logger(__name__)


@tool
async def update_insured_profile(
    runtime: ToolRuntime[ChatRuntimeContext],
    insured_profile: InsuredProfile
) -> Command:
    """保存或更新被保人的年龄、年预算和职业。"""
    profile = insured_profile.model_dump()
    return state_command(runtime, {"insured_profile": profile}, "update success")

@tool
async def get_insured_profile(
    runtime: ToolRuntime[ChatRuntimeContext],
) -> Command:
    """查询被保人的画像，包括年龄、年预算和职业。"""
    profile = runtime.state.get("insured_profile", {})
    # 不更新state，所以直接返回结果即可
    return profile

@tool
async def load_candidates(
    runtime: ToolRuntime[ChatRuntimeContext],
    insurance_category: list[InsuranceCategory] | None = None,
    premium_min: float | None = None,
) -> str | Command:
    """查询候选产品，用于给用户推荐保险产品。
    Args:
        insurance_category: 是保险类型，没有可留空，可选值: medical, critical_illness, accident, life。
        premium_min: 表示产品最低保费的严格上限。
    """
    # 如果没有传保险分类，就四种都查询
    categories = insurance_category or [c.value for c in InsuranceCategory]

    try:
        candidates = await biz_client.list_candidate_products(
            categories,
            premium_min=premium_min,
            limit_per_category=5,
        )

        return state_command(
            runtime,
            {"candidate_products": candidates},
            {"candidate_products": candidates}
        )
    except Exception as exc:
        logger.exception("recommendation_candidates_load_failed")
        return {"error": "候选产品查询失败", "detail": str(exc)}


@tool
async def search_insurance_knowledge(
    query: str,
    runtime: ToolRuntime[ChatRuntimeContext],
    product_id: int,
    top_k: int = 5,
) -> str:
    """检索保险产品条款，返回回答证据和可追溯引用来源。"""

    try:
        # 检索
        evidences: list[RetrievedChunk] = await retriever().search(query, product_id, top_k)

        # 组织检索到的条款内容
        clauses = []  # 给AI，给关键信息即可
        clause_sources = {}  # 给前端，信息要完整
        for e in evidences:
            clauses.append({"source_id": e.source_id, "content": e.content})
            clause_sources[e.source_id] = e

        # 把需要传递给前端的拓展消息写入事件流，这里是条款的引用来源
        writer = get_stream_writer()
        writer({"type": "additional_info", "data": clause_sources})

        return state_command(
            runtime,
            {"additional_info": clause_sources},
            {"clauses": clauses},
        )
    except Exception as exc:
        logger.exception("recommendation_plan_knowledge_search_failed", product_id=product_id)
        return {"error": "保险条款检索失败", "detail": str(exc)}


def state_command(
    runtime: ToolRuntime[ChatRuntimeContext],
    update: dict[str, Any],
    content: Any,
) -> Command:
    """更新 Workflow State，并返回当前 Tool Call 对应的 ToolMessage。"""

    return Command(
        update={
            **update,
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=runtime.tool_call_id or "unknown",
                )
            ],
        }
    )

tools = [
    update_insured_profile,
    get_insured_profile,
    load_candidates,
    search_insurance_knowledge
]