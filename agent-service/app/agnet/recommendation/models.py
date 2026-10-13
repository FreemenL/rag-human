from enum import StrEnum
from operator import add
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field

from app.agnet.models import ParentState


# 保险类别
class InsuranceCategory(StrEnum):
    MEDICAL = "medical"
    CRITICAL_ILLNESS = "critical_illness"
    ACCIDENT = "accident"
    LIFE = "life"

# 被保人信息
class InsuredProfile(BaseModel):
    age: int = Field(default=None, description="被保人年龄")
    budget: float = Field(default=None, description="年缴保费预算")
    occupation: str = Field(default=None, description="职业")


def add_candidate_products(raw: list[dict[str, Any]], news: list[dict[str, Any]]):
    """用于合并候选产品的Reducer"""
    if not raw :
        return news
    merged = {item["id"]: item for item in raw + news}
    return list(merged.values())

class RecommendationPlanState(ParentState):
    """推荐 子agent私有数据"""
    insured_profile: Optional[dict[str, Any]] # 用户画像
    candidate_products: Annotated[list[dict[str, Any]], add_candidate_products] # 候选产品，可以不断累加